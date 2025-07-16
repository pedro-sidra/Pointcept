"""
Sonata v1m1 Base

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from itertools import chain
from packaging import version
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
from timm.layers import trunc_normal_

import pointops
from pointcept.models.sonata.sonata_v1m1_base import Sonata
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.utils.comm import get_world_size, all_gather
from pointcept.utils.scheduler import CosineScheduler


@MODELS.register_module("SonataSculptor-v1m1")
class SonataLikeSculptor(Sonata):
    def __init__(self, **kwargs):
        super(SonataLikeSculptor, self).__init__(**kwargs)
        head_in_channels = kwargs.get("head_in_channels", 32)
        head_hidden_channels = kwargs.get("head_hidden_channels", 32)
        self.sculpt_head = nn.Sequential(
            nn.Linear(head_in_channels, head_hidden_channels),
            nn.GELU(),
            nn.Linear(head_hidden_channels, 2),
        )

    def before_step(self):
        super().before_step()
        self.trainer.comm_info["input_dict"]["mask_params"] = dict(
            mask_size=self.mask_size, mask_ratio=self.mask_ratio
        )

    def forward(self, data_dict, return_point=False):
        if return_point:
            point = self.teacher.backbone(data_dict)
            for _ in range(self.up_cast_level):
                assert "pooling_parent" in point.keys()
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            return dict(point=point)

        # prepare global_point, mask_global_point, local_point
        with torch.no_grad():
            # global_point & masking

            feat = data_dict["global_feat"]
            segment = data_dict["global_segment"]
            coord = data_dict["global_coord"]
            origin_coord = data_dict["global_origin_coord"]
            batch = offset2batch(data_dict["global_offset"])
            grid_size = data_dict["grid_size"][0]

            clean_indexes = segment != 0

            global_point = Point(
                feat=feat[clean_indexes],
                segment=segment[clean_indexes],
                coord=coord[clean_indexes],
                origin_coord=origin_coord[clean_indexes],
                offset=batch2offset(batch[clean_indexes]),
                grid_size=grid_size,
            )

            mask_global_point = Point(
                feat=feat,
                coord=coord,
                origin_coord=origin_coord,
                offset=data_dict["global_offset"],
                grid_size=grid_size,
                mask=~clean_indexes,
            )

            # local point & matching
            clean_indexes = data_dict["local_segment"] != 0
            local_point = Point(
                feat=data_dict["local_feat"][clean_indexes],
                segment=data_dict["local_segment"][clean_indexes],
                coord=data_dict["local_coord"][clean_indexes],
                origin_coord=data_dict["local_origin_coord"][clean_indexes],
                offset=batch2offset(
                    offset2batch(data_dict["local_offset"])[clean_indexes]
                ),
                grid_size=data_dict["grid_size"][0],
            )

            # create result dictionary for return
            result_dict = dict(loss=[])
            # teacher backbone forward (shared with mask and unmask)
            global_point_ = self.teacher.backbone(global_point)
            global_point_ = self.up_cast(global_point_)
            global_feat = global_point_.feat

        if self.sculpt_weight > 0:
            # teacher head forward
            with torch.no_grad():
                global_point_.feat = self.teacher.unmask_head(global_feat)
            # student forward
            local_point_ = self.student.backbone(local_point)
            local_point_ = self.up_cast(local_point_)

            unmask_pred_sim = self.student.unmask_head(local_point_.feat)
            with torch.no_grad():
                principal_view_mask = global_point_.batch % self.num_global_view == 0
                principal_view_batch = (
                    global_point_.batch[principal_view_mask] // self.num_global_view
                )
                match_index = self.match_neighbour(
                    local_point_.origin_coord,
                    local_point_.offset[self.num_local_view - 1 :: self.num_local_view],
                    global_point_.origin_coord[principal_view_mask],
                    batch2offset(principal_view_batch),
                )
                # teacher forward
                unmask_target_sim = self.sinkhorn_knopp(
                    global_point_.feat[principal_view_mask][match_index[:, 1]],
                    self.teacher_temp,
                )
            # loss
            unmask_loss = -torch.sum(
                unmask_target_sim
                * F.log_softmax(
                    unmask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                ),
                dim=-1,
            )
            unmask_loss = torch_scatter.segment_coo(
                unmask_loss,
                index=local_point_.batch[match_index[:, 0]],
                reduce="mean",
            ).mean()
            result_dict["unmask_loss"] = unmask_loss
            result_dict["loss"].append(unmask_loss * self.unmask_loss_weight)

        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            # teacher head forward
            with torch.no_grad():
                global_point_.feat = self.teacher.mask_head(global_feat)
            # student forward
            mask_global_point_ = self.student.backbone(mask_global_point)
            mask_global_point_ = self.up_cast(mask_global_point_)
            mask_pred_sim = self.student.mask_head(mask_global_point_.feat)

            if self.mask_loss_weight > 0:
                with torch.no_grad():
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        global_point_.origin_coord,
                        global_point_.offset,
                    )
                    # teacher forward
                    mask_target_sim = self.sinkhorn_knopp(
                        global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                # loss
                mask_loss = -torch.sum(
                    mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                mask_loss = torch_scatter.segment_coo(
                    mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["mask_loss"] = mask_loss
                result_dict["loss"].append(mask_loss * self.mask_loss_weight)

            if self.roll_mask_loss_weight > 0:
                roll_global_point_ = self.roll_point(global_point_)
                with torch.no_grad():
                    # match index for pred and roll target
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        roll_global_point_.origin_coord,
                        roll_global_point_.offset,
                    )
                    # teacher forward
                    roll_mask_target_sim = self.sinkhorn_knopp(
                        roll_global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                roll_mask_loss = -torch.sum(
                    roll_mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                roll_mask_loss = torch_scatter.segment_coo(
                    roll_mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["roll_mask_loss"] = roll_mask_loss
                result_dict["loss"].append(roll_mask_loss * self.roll_mask_loss_weight)

        result_dict["loss"] = sum(result_dict["loss"])

        if get_world_size() > 1:
            for loss in result_dict.values():
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        return result_dict
