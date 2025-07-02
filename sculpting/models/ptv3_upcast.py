from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    PointTransformerV3,
)
import torch
from pointcept.models.builder import MODELS


@MODELS.register_module("PT-v3m3")
class PointTransformerV3_Upcasting(PointTransformerV3):
    def __init__(
        self,
        up_cast_level=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.up_cast_level = up_cast_level

    def up_cast(self, point):
        for _ in range(self.up_cast_level):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def forward(self, data_dict):
        point = self.up_cast(point=super().forward(data_dict))

        return point
