from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from .det3d_dataset import Det3DDataset
from mmdet3d.structures import LiDARInstance3DBoxes

CLASS_NAMES = [
    "car",
    "truck",
    "trailer",
    "human",
    "reach_stacker",
    "crane",
    "forklift",
    "mast",
    "barrier",
    "sign",
    "machine_other",
    "elevated_container",
]

COLOR_MAP = {
    "car": (106, 0, 228),  # velvet
    "truck": (121, 36, 206),  # violet
    "trailer": (0, 0, 192),  # dark blue
    "human": (255, 82, 82),  # light red
    "reach_stacker": (249, 155, 15),  # soft orange
    "crane": (55, 255, 175),  # menthol green
    "forklift": (240, 240, 128),  # soft yellow
    "barrier": (240, 240, 128),  # soft yellow,
    "mast": (0, 255, 255),  # light blue
    "sign": (255, 255, 0),  # yellow
    "machine_other": (255, 0, 255),  # pink
    "elevated_container": (0, 255, 0),  # green
}


@DATASETS.register_module()
class Fern3dDataset(Det3DDataset):
    r"""Fern3d Dataset.

    This class serves as the API for experiments on the `Fern3d Dataset
    """

    METAINFO = {
        'classes': CLASS_NAMES,
        'palette': [COLOR_MAP[x] for x in CLASS_NAMES],
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 **kwargs):
        assert box_type_3d.lower() in ['lidar']
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)


    def parse_ann_info(self, info: dict) -> dict | None:
        # custom fernride annotation not used in training explicitly
        if 'fern_info' in info.keys():
            scan_path = info['fern_info']['scan_path']
            timestamp = int(info['fern_info']['timestamp'])
            source = str(info["fern_info"]["source"])
            side = -1
            if "left" in source:
                side = 0
            elif "right" in source:
                side = 1
            elif "front" in source:
                side = 2
            scene_id = int(source.split("segments/")[1].split("/")[0])
            del info['fern_info']
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['has_load'] = np.zeros(0, dtype=np.int64)
        
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        ann_info["side"] = side
        ann_info['scene_id'] = scene_id
        ann_info['timestamp'] = timestamp
        return ann_info
