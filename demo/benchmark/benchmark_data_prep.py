import numpy as np
from pathlib import Path
import pickle
import open3d as o3d
import gzip
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

INTENSITY_NORMALIZATION_CONSTANT = 65535.0

def save_pointcloud(xyz: np.ndarray, intensity: np.ndarray, path: Path) -> None:
    """Save pointcloud in KITTI format.
    Args:
        xyz: np.ndarray
        intensity: np.ndarray
        path: Path
    """
    assert xyz.shape[1] == 3
    assert intensity.shape[0] == xyz.shape[0]
    points = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    points[:, :3] = xyz.astype(np.float32)
    points[:, 3] = intensity.astype(np.float32) / INTENSITY_NORMALIZATION_CONSTANT
    with open(path, "wb") as f:
        f.write(points.tobytes())

def ouster_rep_to_ref(value: int):
    """Convert Ouster representation to reflectivity.

    Maps table from here: https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html#calibrated-reflectivity-v2-x

    Args:
        value: int [0-255]
    Returns:
        reflectivity 0 - 86400
    """
    assert 0 <= value <= 255, "expected value in range [0, 255]"
    if value <= 100:
        return value
    factor = int((value - 100) // 16)
    factor = max(0, min(9, factor))
    factor_f = (value - 100) - 16 * factor
    multipliers = [6.25, 12.5, 25, 50, 100, 200, 400, 800, 1600, 3200]
    bases = [100.0]
    for x in multipliers:
        bases.append(bases[-1] + x * 16.0)
    val = bases[factor] + multipliers[factor] * factor_f
    return val

def load_scan(full_path: Path, transform: np.array = np.eye(4)) -> o3d.t.geometry.PointCloud:
    """Load a scan from a file.

    Newer batches have .pkl files with the scan data, older ones have .pcd files. This function will load either.
    Note that pcd files don't contain timestamps. Thus, the timestamps are set to 0.

    Args:
        full_path: Path
        transform: np.array - transformation matrix 4x4
    Returns:
        pcl_t: o3d.t.geometry.PointCloud
    """
    if full_path.suffix == ".pcd":
        if np.linalg.norm(transform - np.eye(4)) > 1e-6:
            raise ValueError("Transform is not supported for PCD files")
        pcl_t = o3d.t.io.read_point_cloud(str(full_path))
        pcl_t.point.timestamps = o3d.core.Tensor(np.zeros((pcl_t.point.positions.shape[0], 1), dtype=np.int64))
        return pcl_t
    elif full_path.suffix == ".pkl" or full_path.suffix == ".gz":
        if full_path.suffix == ".gz":
            with gzip.open(full_path, "rb") as f:
                data = pickle.load(f)
        else:
            data = pickle.load(open(full_path, "rb"))
        positions = data["xyz"].reshape(-1, 3)
        colors = data["ir"].reshape(-1, 1)
        intensities = np.array([ouster_rep_to_ref(x) for x in data["ref"].flatten()]).reshape(-1, 1)
        timestamps = data["timestamp"].reshape(-1, 1)
        assert positions.shape[0] == intensities.shape[0]
        assert colors.shape[0] == intensities.shape[0]
        assert timestamps.shape[0] == intensities.shape[0]
        mask = np.nonzero(np.linalg.norm(positions, axis=1) > 0.1)[0]
        positions = positions[mask, :]
        intensities = intensities[mask, :]
        colors = colors[mask, :]
        timestamps = timestamps[mask, :]
        positions = np.dot(positions, transform[:3, :3].T) + transform[:3, 3]
        colors = np.stack([colors.flatten(), colors.flatten(), colors.flatten()], axis=1).astype(np.float32) / (2**12)
        pcl_t = o3d.t.geometry.PointCloud(
            {
                "positions": o3d.core.Tensor(positions),
                "intensity": o3d.core.Tensor(intensities),
                "colors": o3d.core.Tensor(colors),
                "timestamps": o3d.core.Tensor(timestamps),
            }
        )
        return pcl_t
    raise ValueError(f"Unknown file type {full_path}")     

def get_extrinsics_for_sensor(sensor_name: str, parameter_file: Path) -> np.ndarray:
    """Get extrinsics for a sensor
    Args:
        sensor_name: name of the sensor
        parameter_file: path to the parameter file
    Returns:
        pose_mat: 4x4 transformation matrix to chassis from sensor
    """
    assert sensor_name in ["lidar_front", "lidar_left", "lidar_right"]
    with open(parameter_file) as f:
        params = yaml.safe_load(f)

    if "lidar_poses" in params.keys():
        # new format
        sensor_name_key = {
            "lidar_front": "ouster_front",
            "lidar_left": "ouster_left",
            "lidar_right": "ouster_right",
        }[sensor_name]
        pose_values = params["lidar_poses"][sensor_name_key]
    else:
        pose_values = params[sensor_name]["pose"]

    rotmat = Rotation.from_quat(
        [
            pose_values["rotation_x"],
            pose_values["rotation_y"],
            pose_values["rotation_z"],
            pose_values["rotation_w"],
        ]
    ).as_matrix()
    tvec = np.array(
        [
            pose_values["translation_x"],
            pose_values["translation_y"],
            pose_values["translation_z"],
        ]
    )
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = rotmat
    pose_mat[:3, 3] = tvec
    return pose_mat

def get_transform_to_chassis_from_sensor(prefix: str, dataset_dir: Path):
    with open(dataset_dir / "VERSION") as version_file:
        version = version_file.read().strip()
    if version == "0.0.1":
        return np.eye(4)
    elif version == "0.0.2":
        assert prefix in ["-lidar_front", "-lidar_left", "-lidar_right", ""], f"{prefix} not supported"
        if len(prefix) == 0:
            sensor_name = "lidar_right"
        else:
            sensor_name = prefix[1:]
        return get_extrinsics_for_sensor(sensor_name, dataset_dir / "raw_data/recording_parameter_data.yaml")

if __name__ == "__main__":
    segment_path = Path("/home/tstubler/testing_data/mmdet3d_test/workspace")
    scans_dir = segment_path / "points"
    raw_data_dir = segment_path / "raw_data"
    prefix = "-lidar_right"
    all_pklgz_files = [scan for scan in raw_data_dir.iterdir() if scan.suffix == ".gz"]
    for scan in tqdm(all_pklgz_files, desc="Converting .pkl.gz to .bin"):
        if not scan.suffix == ".gz" or not scan.name.startswith(prefix[1:]):
            continue

        pose_chassis_from_sensor = get_transform_to_chassis_from_sensor(prefix, segment_path)
        
        pcl = load_scan(scan, pose_chassis_from_sensor)
        save_pointcloud(
            pcl.point.positions.numpy(), pcl.point.intensity.numpy().flatten(), scans_dir / f"{scan.name.split('.')[0]}.bin"
        )