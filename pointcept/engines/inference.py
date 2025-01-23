import time
from pointcept.datasets import build_dataset, collate_fn
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
from pypcd.pypcd import pandas_to_pypcd
from pypcd import pypcd

from pointcept.engines.test import TESTERS, TesterBase
from pointcept.utils.logger import get_root_logger
from pointcept.utils.misc import AverageMeter
from pointcept.datasets import collate_fn
import pointcept.utils.comm as comm


@TESTERS.register_module()
class SemSegPredictor(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1

        # Logging setup
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Inference >>>>>>>>>>>>>>>>")

        # To measure inference time
        batch_time = AverageMeter()

        self.model.eval()

        # Creates results folder
        save_path = Path(self.cfg.save_path) / "result"
        save_path.mkdir(parents=True, exist_ok=True)

        # Sync GPUs
        comm.synchronize()

        # Starts inference
        for idx, input_dict in enumerate(
            self.test_loader
        ):  # TODO: Check if we can use tqdm here
            end = time.time()
            input_dict = input_dict[0]  # Assuming batch size is 1

            data_name = input_dict["name"]
            coord = input_dict["coord"]

            # For evaluation purposes, the GT information will also be saved
            # (If data doesn't have GT info, placeholders are created)
            # segment_gt = input_dict["segment"]
            # instance_gt = input_dict["instance"]

            # Set all input data to CUDA
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.model(input_dict)

            output_dict["conf"], output_dict["pred"] = (
                output_dict["seg_logits"].softmax(1).max(1)
            )

            data_export = dict()
            data_export.update(output_dict)
            data_export.update(input_dict)
            data = dict()
            reference_len = len(input_dict["coord"])
            renames = {
                "coord": ["x", "y", "z"],
                "feat": ["r", "g", "b"],
                "segment": "label",
            }
            for key, value in data_export.items():
                if not (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and len(value) == reference_len
                ):
                    print(
                        f"WARNING: {key} not added to output data dict, not tensor or right size"
                    )
                    continue

                value = value.cpu().numpy()
                if value.ndim == 2:
                    for col in range(value.shape[1]):
                        if len(renames.get(key, [])) > col:
                            data[renames.get(key)[col]] = value[:, col]
                        else:
                            data[f"{key}_{col}"] = value[:, col]
                elif value.ndim == 1:
                    if renames.get(key):
                        data[renames.get(key)] = value
                    else:
                        data[key] = value
                else:
                    print(f"WARNING: {key} not added to output data dict, ndim > 2")

            output_pcd = pd.DataFrame(data).astype(np.float32)
            # Updates inference time
            batch_time.update(time.time() - end)

            # Save inference DataFrame to a PCD file
            self.save_inference_to_pcd(str(save_path), data_name, output_pcd)

            # Log test loader inference progression
            logger.info(
                "Inference: {}/{} - {data_name} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) ".format(
                    idx + 1,
                    len(self.test_loader),
                    data_name=data_name,
                    batch_time=batch_time,
                )
            )

        logger.info("<<<<<<<<<<<<<<<<< End Inference <<<<<<<<<<<<<<<<<")

    def save_inference_to_pcd(self, output_dir, data_name, df):
        if "r" in df.columns and "g" in df.columns and "b" in df.columns:
            colors = df[["r", "g", "b"]].to_numpy()
            colors = (colors + 1) / 2 * 255
            df["rgb"] = pypcd.encode_rgb_for_pcl(colors.astype(np.uint8))
            df.drop(columns=["r", "g", "b"], inplace=True)

        pandas_to_pypcd(df).save_pcd(
            f"{output_dir}/{data_name}.pcd", compression="binary_compressed"
        )

    @staticmethod
    def collate_fn(batch):
        return batch

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader
