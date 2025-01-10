from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from mmengine.structures import InstanceData
from mmengine import Config
from mmdet3d.apis.inferencers.base_3d_inferencer import Base3DInferencer
from mmengine.dataset import Compose
from mmengine.infer.infer import ModelType

from tqdm import tqdm
import pickle
import click
from pathlib import Path
import torch

from mmdet3d.structures import Box3DMode, Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.utils import ConfigType
from demo.data_models import gen_simple_cuboid_from_bbox

import open3d as o3d

InputType = Union[str, np.ndarray]
InstanceList = List[InstanceData]
PredType = Union[InstanceData, InstanceList]
InputsType = Union[InputType, Sequence[InputType]]


COLOR_MAP = {
    "car": (106, 0, 228),  # velvet
    "truck": (121, 36, 206),  # violet
    "trailer": (0, 0, 192),  # dark blue
    "human": (255, 82, 82),  # light red
    "reach_stacker": (249, 155, 15),  # soft orange
    "crane": (55, 255, 175),  # menthol green
    "forklift": (240, 240, 128),  # soft yellow
    "barrier": (240, 0, 16),  #
    "mast": (0, 255, 255),  # light blue
    "sign": (255, 255, 0),  # yellow
}


class CustomModelRunner(Base3DInferencer):
    def __init__(self,
                model: Union[ModelType, str, None] = None,
                weights: Optional[str] = None,
                device: Optional[str] = None,
                scope: str = 'mmdet3d',
                palette: str = 'none'
    ):
        super().__init__(model=model, weights=weights, device=device, scope=scope, palette=palette)


    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        load_point_idx = self._get_transform_idx(pipeline_cfg,
                                                 'LoadPointsFromFile')
        if load_point_idx == -1:
            raise ValueError(
                'LoadPointsFromFile is not found in the test pipeline')

        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type, self.load_dim = load_cfg['coord_type'], load_cfg[
            'load_dim']
        self.use_dim = list(range(load_cfg['use_dim'])) if isinstance(
            load_cfg['use_dim'], int) else load_cfg['use_dim']

        pipeline_cfg[load_point_idx]['type'] = 'LidarDet3DInferencerLoader'
        return Compose(pipeline_cfg)


    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = -1,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '') -> Union[List[np.ndarray], None]:
        return None
    

    def run_on_sample(self, sample: np.ndarray):
        assert sample.shape[1] == 4
        torch_sample = {'inputs':{'points': [torch.from_numpy(sample)]}}
        meta_info = {
            "pcd_scale_factor": 1.0,
            'box_type_3d':  LiDARInstance3DBoxes,
            'box_mode_3d': Box3DMode.LIDAR,
            'axis_align_matrix': np.eye(4),
            'pcd_vertical_flip': False,
            'pcd_horizontal_flip': False,
            'lidar_path': '',
            'flip': False
            }
        data_sample = Det3DDataSample(metainfo=meta_info)
        forward_kwargs = dict()
        data_sample = Det3DDataSample(metainfo=meta_info)
        result = self.forward({'inputs':torch_sample['inputs'],
                                'data_samples':[data_sample]
                                }, **forward_kwargs)
        predictions = [x.pred_instances_3d for x in result]
        assert len(predictions) == 1, "should be only 1, since input == 1"
        bboxes = predictions[0].bboxes_3d.cpu().numpy()
        labels = predictions[0].labels_3d.cpu().numpy()
        scores = predictions[0].scores_3d.cpu().numpy()
        return {"bboxes": bboxes, "labels": labels, "scores": scores}


def normalize(data: np.ndarray, percentile: float = 0.05):
    """Normalize and clamp data for better color mapping.

    This is a utility function used ONLY for the purpose of 2D image
    visualization. The resulting values are not fully reversible because the
    final clipping step discards values outside of [0, 1].

    Args:
        data: array of data to be transformed for visualization
        percentile: values in the bottom/top percentile are clamped to 0 and 1

    Returns:
        An array of doubles with the same shape as ``image`` with values
        normalized to the range [0, 1].
    """
    min_val = np.percentile(data, 100 * percentile)
    max_val = np.percentile(data, 100 * (1 - percentile))
    # to protect from division by zero
    spread = max(max_val - min_val, 1)
    field_res = (data.astype(np.float64) - min_val) / spread
    return field_res.clip(0, 1.0)


def make_pointcloud_object(point_cloud_data: np.ndarray):
    assert point_cloud_data.shape[1] == 4
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
    gray = normalize(point_cloud_data[:, 3])
    colors = np.stack([gray, gray, gray], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def run_test_inference(dataset_path: Path, output_path: Path, model_path: Path, overwrite: bool = False):
    # Check if benchmark artifacts already exist:
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    output_file = output_path / "benchmark_artifacts.pkl"
    if output_file.exists():
        if overwrite:
            output_file.unlink()
        else:
            print("Artifacts already exist. Use --overwrite to regenerate.")
            return
    
    # Load the model and weights:
    model = model_path / "pointpillars_fern3d_dynamic.py"
    weights = [x for x in model_path.iterdir() if x.suffix == ".pth"][0]

    # Initialize Inference Runner:
    inferencer = CustomModelRunner(model=str(model), weights=str(weights), device='cuda:0')
    
    # Initialize benchmark dataset:
    dataset = sorted([f for f in dataset_path.iterdir() if f.is_file() and f.suffix == ".bin"])
    
    # Initialize class names:
    class_names = [x for x in COLOR_MAP.keys()]
    
    # Run inference for all samples in benchmark dataset:
    inference_results = dict()
    for sample_pcd_file in tqdm(dataset, desc="Running inference on all dataset samples"):
        file_name = sample_pcd_file.stem
        data = np.fromfile(sample_pcd_file, dtype=np.float32)
        data = data.reshape(-1, 4)
        pcd = make_pointcloud_object(data)
        pcd.estimate_normals()
        inference_result = inferencer.run_on_sample(data)
        boxes = [gen_simple_cuboid_from_bbox(x) for x in inference_result["bboxes"]]
        for box, label in zip(boxes, inference_result["labels"]):
            box.class_name = class_names[label]
        inference_results[file_name] = boxes

    # Save artifacts as .pkl:
    pickle.dump(
        {
            "model_name": model_path.stem, # or version (in order to check, data from same model basis is compared against each other)
            "results": inference_results,
        },
        open(output_file, "wb"),
    )

@click.command()
@click.argument("model_path", type=click.types.Path(path_type=Path, dir_okay=True, exists=True))
@click.option("--output_path", type=click.types.Path(path_type=Path, dir_okay=True), required=True)
@click.option("--dataset_path", type=click.types.Path(path_type=Path, dir_okay=True), required=True)
@click.option("--overwrite", is_flag=True)
def cli(dataset_path: Path, output_path: Path, model_path: Path, overwrite: bool):
    run_test_inference(dataset_path, output_path, model_path, overwrite)
    
if __name__ == "__main__":
    cli()