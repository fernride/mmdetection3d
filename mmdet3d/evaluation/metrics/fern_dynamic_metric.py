import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode,
                                LiDARInstance3DBoxes)


def bounding_box_to_corners(bbox):
    """
    Convert bounding box parameters to corner points.
    """
    x, y, z, length, width, height, yaw = bbox
    corners = np.array([
        [length/2, width/2, 0.0],        # Front left bottom 
        [length/2, -width/2, 0.0],       # Front right bottom 
        [length/2, -width/2, height],    # Front right top 
        [length/2, width/2, height],     # Front left top
        [-length/2, width/2, 0.0],       # Back left bottom
        [-length/2, -width/2, 0.0],      # Back right bottom
        [-length/2, -width/2, height],   # Back right top
        [-length/2, width/2, height]     # Back left top
    ])
    # Rotation matrix
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    transformed_corners = np.dot(corners, R.T) + np.array([x, y, z])
    return transformed_corners


def shortest_distance_between_bboxes(bbox1, bbox2):
    """
    Compute the shortest distance between two bounding boxes.
    """
    corners1 = bounding_box_to_corners(bbox1)
    corners2 = bounding_box_to_corners(bbox2)
    min_distance = np.inf

    # Compute distance between all pairs of points
    for corner1 in corners1:
        for corner2 in corners2:
            distance = np.linalg.norm(corner1 - corner2)
            if distance < min_distance:
                min_distance = distance

    return min_distance



def test_bbs_against_bb(bbs, bbs_classes, bb, bb_class):
    """Test if a list of bounding boxes intersect with a single bounding box.
    
    """
    assert len(bbs) == len(bbs_classes)
    bb_this = LiDARInstance3DBoxes(tensor=bb)
    max_overlap = None
    max_overlap_index = -1


    for index, (other_bb, other_bb_class) in enumerate(zip(bbs, bbs_classes)):
        bb_other = LiDARInstance3DBoxes(tensor=other_bb)
        overlap = bb_this.overlaps(bb_other)
        if max_overlap is None or overlap > max_overlap:
            max_overlap = overlap
            max_overlap_index = index
    
    if max_overlap_index == -1:
        return False, -1


def calculate_vru_detection_quality():
    pass

def calculate_vehicle_detection_quality():
    pass 

def calculate_ghosts(pred_bbs, pred_classes, gt_bbs, gt_classes):

    pass
    
def calculate_crane_detection_quality():
    pass


@METRICS.register_module()
class FernDynamicMetric(BaseMetric):
    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Fern Dynamic Metric'
        super(FernDynamicMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.format_only = format_only
        self.backend_args = backend_args

        allowed_metrics = ['bbox', 'mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'bbox', 'img_bbox', "
                               f'but got {metric}.')
            


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        print(self.dataset_meta['classes'])
        
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances_3d']
            pred_scores = pred['scores_3d'].cpu().numpy()
            pred_bboxes = pred['bboxes_3d'].cpu().numpy()
            pred_labels = pred['labels_3d'].cpu().numpy()

            gt_ann = data_sample['eval_ann_info']
            gt_bboxes = gt_ann['gt_bboxes_3d'].cpu().numpy()
            gt_labels = gt_ann['gt_labels_3d'].cpu().numpy()
            num_points = gt_ann['num_lidar_pts'].cpu().numpy()
            # here need to do optimal assignment



            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)


    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # load annotations
        #pkl_infos = load(self.ann_file, backend_args=self.backend_args)
        #self.data_infos = self.convert_annos_to_kitti_annos(pkl_infos)
        #result_dict, tmp_dir = self.format_results(
        #    results,
        #    pklfile_prefix=self.pklfile_prefix,
        #    submission_prefix=self.submission_prefix,
        #    classes=self.classes)

        metric_dict = {}

        #if self.format_only:
        logger.info(f'results are not saved in')
        return metric_dict

