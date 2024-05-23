from mmcv.transforms.base import BaseTransform
from mmengine.registry import TRANSFORMS
from mmengine.structures import InstanceData

from mmdet3d.datasets import Fern3dDataset
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


def _generate_nus_dataset_config():
    data_root = 'tests/data/lyft'
    ann_file = 'lyft_infos.pkl'
    classes = [
        'car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
        'motorcycle', 'bicycle', 'pedestrian', 'animal'
    ]
    if 'Identity' not in TRANSFORMS:

        @TRANSFORMS.register_module()
        class Identity(BaseTransform):

            def transform(self, info):
                packed_input = dict(data_samples=Det3DDataSample())
                if 'ann_info' in info:
                    packed_input[
                        'data_samples'].gt_instances_3d = InstanceData()
                    packed_input[
                        'data_samples'].gt_instances_3d.labels_3d = info[
                            'ann_info']['gt_labels_3d']
                return packed_input

    pipeline = [
        dict(type='Identity'),
    ]
    modality = dict(use_lidar=True, use_camera=False)
    data_prefix = dict(pts='lidar', img='', sweeps='sweeps/LIDAR_TOP')
    return data_root, ann_file, classes, data_prefix, pipeline, modality



