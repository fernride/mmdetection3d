import tempfile

from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.models.layers import box3d_multiclass_nms

from scipy.optimize import linear_sum_assignment


from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode,
                                LiDARInstance3DBoxes)



do_once = True

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
    

def get_corners(rect):
    cx, cy, cz, l, w, h, yaw = rect
    half_l, half_w = l / 2, w / 2
    corners = np.array([
        [-half_l, -half_w],
        [-half_l,  half_w],
        [ half_l,  half_w],
        [ half_l, -half_w]
    ])
    # Rotation matrix for yaw
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw,  cos_yaw]
    ])
    # Rotate and translate corners
    rotated_corners = (rotation_matrix @ corners.T).T
    translated_corners = rotated_corners + np.array([cx, cy])
    return translated_corners


def get_ref_corners(rect):
    cx, cy, cz, l, w, h, yaw = rect
    half_l, half_w = l / 2, w / 2
    corners = np.array([
        [-half_l, -half_w],
        [-half_l,  half_w],
        [ half_l,  half_w],
        [ half_l, -half_w]
    ])
    return corners


def get_distance_to_ref_point(point, bbox: np.ndarray):
    assert len(bbox) == 7
    cx = bbox[0]
    cy = bbox[1]
    cz = bbox[2]
    l = bbox[3]
    w = bbox[4]
    h = bbox[5]
    yaw = bbox[6]
    # Translate the point into the rectangle's frame
    #point = np.array([px, py, pz])
    center = np.array([cx, cy, cz])
    translated_point = point - center

    # Rotation matrix for yaw (around z-axis)
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])
    
    # Rotate the point into the rectangle's local coordinate system
    local_point = rotation_matrix @ translated_point

    # Clamp the point to the rectangle bounds
    half_lengths = np.array([l / 2, w / 2, h / 2])
    clamped_point = np.maximum(-half_lengths, np.minimum(local_point, half_lengths))
    
    # Compute the closest point in global coordinates
    closest_local_point = clamped_point
    closest_global_point = np.linalg.inv(rotation_matrix) @ closest_local_point + center

    # Compute the distance
    distance = np.linalg.norm(point - closest_global_point)
    return distance


def get_distance_from_box_to_boxes(ref_bbox, test_bbox):
    test_corners = get_corners(test_bbox)
    test_h = test_bbox[2]
    test_corners = np.vstack([test_corners.T, test_h * np.ones(4)]).T
    distances = np.zeros(4)
    for ind in range(4):
        distances[ind] = get_distance_to_ref_point(test_corners[ind,:], ref_bbox)
    return np.min(distances)


def compute_iou_transformed(ref_bbox, test_bbox):
    # Transform `test_bbox` into the local frame of `ref_bbox`
    def transform_to_local_frame(ref_bbox, test_bbox):
        cx_ref, cy_ref, cz_ref, _, _, _, yaw_ref = ref_bbox
        cx_test, cy_test, cz_test, l_test, w_test, h_test, yaw_test = test_bbox

        # Translate the test_bbox center relative to ref_bbox center
        translated_center = np.array([cx_test - cx_ref, cy_test - cy_ref])

        # Rotation matrix to align with ref_bbox frame
        cos_yaw_ref, sin_yaw_ref = np.cos(-yaw_ref), np.sin(-yaw_ref)
        rotation_matrix = np.array([
            [cos_yaw_ref, -sin_yaw_ref],
            [sin_yaw_ref,  cos_yaw_ref]
        ])

        # Rotate the test_bbox center
        local_center = rotation_matrix @ translated_center

        # Update test_bbox parameters in local frame
        local_yaw = yaw_test - yaw_ref  # Relative yaw
        return [local_center[0], local_center[1], cz_test, l_test, w_test, h_test, local_yaw]

    # Calculate axis-aligned intersection area
    def calculate_intersection_area(corners1, corners2):
        # Bounding box ranges in x and y
        min_x1, min_y1 = np.min(corners1, axis=0)
        max_x1, max_y1 = np.max(corners1, axis=0)
        min_x2, min_y2 = np.min(corners2, axis=0)
        max_x2, max_y2 = np.max(corners2, axis=0)

        # Compute the overlap in x and y
        overlap_x = max(0, min(max_x1, max_x2) - max(min_x1, min_x2))
        overlap_y = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))

        # Intersection area
        return overlap_x * overlap_y

    # Get the corners of ref_bbox (axis-aligned in its own frame)
    ref_corners = get_ref_corners(ref_bbox)
    # Transform test_bbox to ref_bbox frame and get its corners
    transformed_test_bbox = transform_to_local_frame(ref_bbox, test_bbox)
    test_corners = get_corners(transformed_test_bbox)
    
    # Calculate intersection area
    intersection_area = calculate_intersection_area(ref_corners, test_corners)

    # Calculate areas of ref_bbox and test_bbox
    ref_area = (ref_bbox[3] * ref_bbox[4])  # length * width
    test_area = (test_bbox[3] * test_bbox[4])  # length * width

    # Union area
    union_area = ref_area + test_area - intersection_area

    # IoU
    iou = intersection_area / (union_area + 1e-6)
    return iou


def optimal_assignment_iou(iou_matrix):
    """
    Perform Hungarian matching to find the optimal assignment based on IoU matrix.

    Parameters:
    - iou_matrix (np.ndarray): A matrix of shape (N, M) where values represent IoU scores 
                                between N ground-truth boxes and M predicted boxes.

    Returns:
    - matched_indices (list of tuples): List of (ground_truth_index, prediction_index) pairs.
    """
    # Convert IoU matrix to cost matrix (higher IoU = lower cost for matching)
    cost_matrix = 1 - iou_matrix

    # Solve the assignment problem using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Combine indices into pairs
    matched_indices = list(zip(row_indices, col_indices))
    
    return matched_indices


def make_optimal_assignment(ref_boxes, test_boxes):
    min_long_distance_match = 1
    default_cost_value = 1000
    assert default_cost_value > min_long_distance_match

    iou_matrix = -default_cost_value * np.ones([len(ref_boxes), len(test_boxes)])

    for ref_ind in range(len(ref_boxes)):
        for test_ind in range(len(test_boxes)):
            iou_value = compute_iou_transformed(ref_boxes[ref_ind], test_boxes[test_ind])
            distance = get_distance_from_box_to_boxes(ref_boxes[ref_ind], test_boxes[test_ind])
            if iou_value < 0.00001 and distance < min_long_distance_match:
                iou_matrix[ref_ind, test_ind] = -distance
            elif iou_value > 0.00001:
                iou_matrix[ref_ind, test_ind] = iou_value
    
    raw_assignment = optimal_assignment_iou(iou_matrix)
    true_positives_matches = []
    for row in raw_assignment:
        if iou_matrix[row[0], row[1]] < -min_long_distance_match:
            continue
        true_positives_matches.append([row[0], row[1]])
    return true_positives_matches


class EvalAccumulator:
    def __init__(self, detection_bins=[0, 10, 25, 50, 70], num_classes=10):
        self._detection_bins = np.array(detection_bins)
        self._sensor_point = np.array([5.2, 0, 1.0])
        num_bins = self._detection_bins.shape[0]-1
        self._tp_cnt_by_range = np.zeros([num_bins, num_classes], dtype=np.int64)
        self._fp_cnt_by_range = np.zeros([num_bins, num_classes], dtype=np.int64)
        self._fn_cnt_by_range = np.zeros([num_bins, num_classes], dtype=np.int64)
        # TODO replace with dict
        self.sensor_point = np.array([5.2, 0, 1.0])

    def process_tp(self, tp_gt_boxes, tp_pred_boxes, gt_boxes, pred_boxes, gt_labels, pred_labels, gt_ranges_to_bins):
        for box_ind in tp_gt_boxes:
            range_ind = gt_ranges_to_bins[box_ind]
            self._tp_cnt_by_range[range_ind, gt_labels[box_ind]] += 1
    
    def print_stat(self):
        print("TP\n", np.sum(self._tp_cnt_by_range, axis=1))
        print("FP\n", np.sum(self._fp_cnt_by_range, axis=1))
        print("FN\n", np.sum(self._fn_cnt_by_range, axis=1))
        
    def range_indices(self, boxes):
        box_ranges = np.array([get_distance_to_ref_point(self.sensor_point, box) for box in boxes])
        box_ranges_to_bins = np.zeros(len(box_ranges), dtype=np.int32)
        for ind in range(self._detection_bins.shape[0]-1):
            box_ranges_to_bins = np.maximum(box_ranges_to_bins, ind * (box_ranges > self._detection_bins[ind]))
        return box_ranges_to_bins
    
    def process_fp(self, fp_pred_boxes, pred_boxes, pred_labels, pred_ranges_to_bins):
        for box_ind in fp_pred_boxes:
            range_ind = pred_ranges_to_bins[box_ind]
            self._fp_cnt_by_range[range_ind, pred_labels[box_ind]] += 1
        
    def process_fn(self, fn_gt_boxes, gt_boxes, gt_labels, gt_ranges_to_bins):
        for box_ind in fn_gt_boxes:
            range_ind = gt_ranges_to_bins[box_ind]
            self._fn_cnt_by_range[range_ind, gt_labels[box_ind]] += 1
    
    def add_items(self, gt_boxes, gt_labels, pred_boxes, pred_labels):
        gt_ranges_to_bins = self.range_indices(gt_boxes)
        pred_ranges_to_bins = self.range_indices(pred_boxes)
        
        class_unaware_tp = make_optimal_assignment(gt_boxes, pred_boxes)
        tp_gt_boxes = [x[0] for x in class_unaware_tp]
        tp_pred_boxes = [x[1] for x in class_unaware_tp]
        fn_gt_boxes = []
        fp_pred_boxes = []
        for box_ind in range(len(gt_boxes)):
            if box_ind not in tp_gt_boxes:
                fn_gt_boxes.append(box_ind)
            else:
                tp_gt_boxes.append(box_ind)
        for box_ind in range(len(pred_boxes)):
            if box_ind not in tp_pred_boxes:
                fp_pred_boxes.append(box_ind)
                
                
        self.process_tp(tp_gt_boxes, tp_pred_boxes, gt_boxes, pred_boxes, gt_labels, pred_labels, gt_ranges_to_bins)
        self.process_fn(fn_gt_boxes, gt_boxes, gt_labels, gt_ranges_to_bins)
        self.process_fp(fp_pred_boxes, pred_boxes, pred_labels, pred_ranges_to_bins)



def calculate_stat_over_classes(class_indices, eval_acc: EvalAccumulator, metric_dict, prefix_str=""):
    tp_cnt_by_range = np.sum(eval_acc._tp_cnt_by_range[:, class_indices], axis=1)
    fp_cnt_by_range = np.sum(eval_acc._fp_cnt_by_range[:, class_indices], axis=1)
    fn_cnt_by_range = np.sum(eval_acc._fn_cnt_by_range[:, class_indices], axis=1)
    precision_by_range = tp_cnt_by_range / (tp_cnt_by_range + fp_cnt_by_range + 1e-6)
    recall_by_range = tp_cnt_by_range / (tp_cnt_by_range + fn_cnt_by_range + 1e-6)
    ghosting_by_range = fp_cnt_by_range / (tp_cnt_by_range + fp_cnt_by_range + 1e-6)
    f1_by_range = 2 * precision_by_range * recall_by_range / (precision_by_range + recall_by_range + 1e-6)
    for ind, (range_value_start, range_value_end) in enumerate(zip(eval_acc._detection_bins[0:-1], eval_acc._detection_bins[1:])):
        range_str = f"{range_value_start}-{range_value_end}"
        metric_dict[f"{prefix_str}_precision_{range_str}"] = precision_by_range[ind]
        metric_dict[f"{prefix_str}_recall_{range_str}"] = recall_by_range[ind]
        metric_dict[f"{prefix_str}_f1_{range_str}"] = f1_by_range[ind]
        metric_dict[f"{prefix_str}_ghosting_{range_str}"] = ghosting_by_range[ind]
    return metric_dict



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
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances_3d']
            pred_scores = pred['scores_3d'].cpu().numpy()
            pred_bboxes = pred['bboxes_3d'].cpu().numpy()
            pred_labels = pred['labels_3d'].cpu().numpy()

            gt_ann = data_sample['eval_ann_info']
            gt_bboxes = gt_ann['gt_bboxes_3d'].cpu().numpy()
            gt_labels = gt_ann['gt_labels_3d']
            track_ids = gt_ann['track_id']
            num_lidar_points = gt_ann['num_lidar_pts']
            timestamp = gt_ann['timestamp']
            side = gt_ann['side']
            scene_id = gt_ann['scene_id']
            # here need to do optimal assignment

            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            result['pred_scores'] = pred_scores
            result['pred_bboxes'] = pred_bboxes
            result['pred_labels'] = pred_labels
            result['gt_bboxes'] = gt_bboxes
            result['gt_labels'] = gt_labels
            result['track_ids'] = track_ids
            result['num_lidar_points'] = num_lidar_points
            result['side'] = side
            result['scene_id'] = scene_id
            result['timestamp'] = timestamp

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
        self.classes : list[str] = self.dataset_meta['classes']
        save_raw = False 
        if save_raw:
            import pickle

            with open("/tmp/test_eval_data.pkl", "wb") as f:
                pickle.dump(results, f)
                print("Saved example data for evaluation")

        eval_acc = EvalAccumulator(num_classes=len(self.classes))
        for frame in results:
            pred_bboxes = frame['pred_bboxes']
            pred_labels = frame['pred_labels']
            gt_bboxes = frame['gt_bboxes']
            gt_labels = frame['gt_labels']
            eval_acc.add_items(gt_bboxes, gt_labels, pred_bboxes, pred_labels)
        
        metric_dict = {}
        dynamic_classes = ["truck", "trailer", "car", "reach_stacker", "forklift", "machine_other"]
        dynamic_classes_indices = [self.classes.index(x) for x in dynamic_classes]
        calculate_stat_over_classes(dynamic_classes_indices, eval_acc, metric_dict, "dynamic")
        static_classes = ["mast"]
        static_classes_indices = [self.classes.index(x) for x in static_classes]
        calculate_stat_over_classes(static_classes_indices, eval_acc, metric_dict, "static")
        if self.format_only:
            logger.info(f'results are not saved')
        return metric_dict

