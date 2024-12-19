from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from mmengine.structures import InstanceData
from mmengine import Config
from mmdet3d.apis.inferencers.base_3d_inferencer import Base3DInferencer
from mmengine.dataset import Compose
from mmengine.infer.infer import ModelType

import click
from pathlib import Path
import torch

from mmdet3d.structures import Box3DMode, Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.utils import ConfigType
from demo.data_models import SimpleCuboid, o3d_bbox_from_cuboid, gen_simple_cuboid_from_bbox

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

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


def dataset_accessor():
    samples_dir = Path("/media/omuratov/bigdata/datasets/fern3d_v0_tiny/points/")
    files = [f for f in samples_dir.iterdir() if f.is_file() and f.suffix == ".bin"]
    for f in files:
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(-1, 4)
        sample = {'inputs':{'points': [torch.from_numpy(data)]}}
        yield sample

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


class Det3dViz:
    def __init__(self, width: int, height: int, dataset_folder: Path, inference: CustomModelRunner):
        self.window = gui.Application.instance.create_window("Det3dViz", width, height)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)
        self._dataset = sorted([f for f in dataset_folder.iterdir() if f.is_file() and f.suffix == ".bin"])
        self._inference = inference
        #self._class_names = ['car', 'truck', 'trailer', 'human', 'reach_stacker']
        self._class_names = [x for x in COLOR_MAP.keys()]
        # should be sourced from model config
        #self._class_names = ["mast", "barrier", "sign"]

        # side panel
        em = self.window.theme.font_size
        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # material to show point clouds
        self.mp_material = rendering.MaterialRecord()
        self.mp_material.shader = "defaultUnlit"
        self.mp_material.point_size = 1.5

        # material to show bounding boxes
        self.bb_material = dict()
        for class_name in COLOR_MAP.keys():
            self.bb_material[class_name] = rendering.MaterialRecord()
            self.bb_material[class_name].shader = "unlitSolidColor"
            self.bb_material[class_name].base_color = [
                max(0.0, min(1.0, COLOR_MAP[class_name][0] / 255.0)),
                max(0.0, min(1.0, COLOR_MAP[class_name][1] / 255.0)),
                max(0.0, min(1.0, COLOR_MAP[class_name][2] / 255.0)),
                1.0,
            ]
            self.bb_material[class_name].line_width = 2.0

        # samples list
        self._samples_list = gui.ListView()
        self._settings_panel.add_child(gui.Label("Samples"))
        self._settings_panel.add_child(self._samples_list)
        self._samples_list.set_on_selection_changed(self._on_samples_selection_changed)
        self._samples = list()

        # layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        self.window.add_child(self._settings_panel)

        # add keycallback
        def on_key(event):
            if event.type == gui.KeyEvent.Type.UP:
                if event.key == gui.KeyName.R:
                    self.reset_camera()
                elif event.key == gui.KeyName.DOWN or event.key == gui.KeyName.RIGHT:
                    self._select_next()
                elif event.key == gui.KeyName.UP or event.key == gui.KeyName.LEFT:
                    self._select_prev()
                elif event.key == gui.KeyName.Q:
                    self.on_exit()

        self.window.set_on_key(on_key)
        self._populate_list()
        self._changes = False

    def on_exit(self, force=False):
        if self._changes and not force:
            print("unsaved changes")
        else:
            self.window.close()

    def _select_prev(self):
        if self._samples_list.selected_index == -1:
            return
        if self._samples_list.selected_index > 0:
            self._samples_list.selected_index -= 1
            value = self._samples[self._samples_list.selected_index]
            self._on_samples_selection_changed(value, False)

    def _select_next(self):
        if self._samples_list.selected_index == -1:
            return
        if self._samples_list.selected_index < len(self._dataset) - 1:
            self._samples_list.selected_index += 1
            value = self._samples[self._samples_list.selected_index]
            self._on_samples_selection_changed(value, False)

    def _on_samples_selection_changed(self, value, is_double_click):
        if not is_double_click:
            self._load_frame(self._samples_list.selected_index)

    def _draw_bbox(self, bboxes):
        for count, bbox in enumerate(bboxes):
            self._scene.scene.add_geometry(f"bbox_{count}", o3d_bbox_from_cuboid(bbox), self.bb_material[bbox.class_name])

    def _load_frame(self, idx):
        self._scene.scene.clear_geometry()
        f = self._dataset[idx]
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(-1, 4)
        pcd = make_pointcloud_object(data)
        pcd.estimate_normals()
        self._scene.scene.add_geometry(f"{idx}", pcd, self.mp_material)
        inference_result = self._inference.run_on_sample(data)
        boxes = [gen_simple_cuboid_from_bbox(x) for x in inference_result["bboxes"]]
        for box, label in zip(boxes, inference_result["labels"]):
            box.class_name = self._class_names[label]
        #print(boxes)
        self._draw_bbox(boxes)

    def _populate_list(self):
        c_selected_index = self._samples_list.selected_index
        self._samples = [x.stem for x in self._dataset]
        self._samples_list.set_items(self._samples)
        self._samples_list.selected_index = c_selected_index

    def _on_layout(self, layout_context):
        """Layout the scene and the settings panel."""
        r = self.window.content_rect
        self._scene.frame = r
        width = 15 * layout_context.theme.font_size
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, r.height)

    def reset_camera(self):
        """Reset the camera to view all the trajectories and perimeters."""
        min_val = [-50.0, -50.0, -0.5]
        max_val = [70.0, 70.0, 10.0]
        bounds = o3d.geometry.AxisAlignedBoundingBox(min_val, max_val)
        self._scene.setup_camera(60.0, bounds, bounds.get_center())


def cli():
    #model_path = Path("/home/omuratov/workspace/fern_ml/output/fernnet/fernnet_human_500")
    model_path = Path("/home/omuratov/workspace/fern_ml/output/fernnet/fern3d_dynamic_500_2024_09_12")
    #model_path = Path("/home/omuratov/workspace/fern_ml/output/fernnet/fern3d_static_2024_11_27_4")
    model_path = Path("/home/omuratov/workspace/fern_ml/output/fernnet/fern3d_b1-b7-9000_v3/")
    model = model_path/"pointpillars_fern3d_dynamic.py"
    #model = model_path/"pp_fern3d_static_only.py"
    weights = [x for x in model_path.iterdir() if x.suffix == ".pth"][0]
    #weights = Path("/home/omuratov/workspace/fern_ml/output/fernnet/fernnet_ped_500")/"epoch_500.pth"
    #model = Path("/home/omuratov/workspace/fern_ml/output/fernnet/fernnet_ped_500")/"pointpillars_fern3d_dynamic.py"
    inferencer = CustomModelRunner(model=str(model), weights=str(weights), device='cuda:0')
    samples_dir = Path("/media/omuratov/bigdata/datasets/fern3d_v0_tiny/points/")
    samples_dir = Path("/media/omuratov/bigdata/datasets/fern3d_b0_b3_filtered/points/")
    #samples_dir = Path("/media/omuratov/bigdata/datasets/fern3d_b1-b7_all/points/")
    gui.Application.instance.initialize()
    app = Det3dViz(width=1200, height=800, dataset_folder=samples_dir, inference=inferencer)
    app.reset_camera()
    gui.Application.instance.run()

if __name__ == "__main__":
    cli()