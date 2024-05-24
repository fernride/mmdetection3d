import numpy as np

from mmcv.transforms.base import BaseTransform
from mmengine.registry import TRANSFORMS
from mmengine.structures import InstanceData

from mmdet3d.datasets import Fern3dDataset
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


def _generate_dataset_config():
    data_root = '/home/omuratov/bigdata/datasets/fern3d_v0'
    ann_file = 'fern3d_test.pkl'
    classes = ['truck', "car", "human"]
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


def test_getitem():
    np.random.seed(0)

    data_root, ann_file, classes, data_prefix, pipeline, modality = \
        _generate_dataset_config()

    dataset = Fern3dDataset(
        data_root,
        ann_file,
        data_prefix=data_prefix,
        pipeline=pipeline,
        metainfo=dict(classes=classes),
        modality=modality)

    dataset.prepare_data(0)
    input_dict = dataset.get_data_info(0)

    assert data_prefix['pts'] in input_dict['lidar_points']['lidar_path']
    assert data_root in input_dict['lidar_points']['lidar_path']

    ann_info = dataset.parse_ann_info(input_dict)

    assert 'gt_labels_3d' in ann_info
    assert ann_info['gt_labels_3d'].dtype == np.int64

if __name__ == '__main__':
    test_getitem()