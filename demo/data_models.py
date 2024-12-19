import logging

import numpy as np
import open3d as o3d
from open3d import geometry
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class SimpleCuboid(BaseModel):
    """Simple cuboid class for 3D bounding boxes

    Convention similar to MMDetection3d bottom centered.
    """

    x: float  # x-coordinate of the center of the cuboid
    y: float  # y-coordinate of the center of the cuboid
    z: float  # z-coordinate of the bottom of the cuboid
    length: float  # length of the cuboid, aka x axis dimension
    width: float  # width of the cuboid, aka y axis dimension
    height: float  # height of the cuboid, aka z axis dimension
    rx: float  # rotation around x axis
    ry: float  # rotation around y axis
    rz: float  # rotation around z axis
    cuboid_id: int  # unique id for the cuboid
    track_id: int | None = None  # unique id for the track
    class_name: str | None = None  # class name of the cuboid

    def to_mmdet_bbox_str(self):
        """Convert the cuboid to a string that can be used in MMDetection3D bbox format"""
        return f"{self.x:0.3f} {self.y:0.3f} {self.z:0.3f} {self.length:0.3f} {self.width:0.3f} {self.height:0.3f} {self.rz:0.3f} {self.class_name}"

    def transform(self, rotmat, tvec):
        """Transform the cuboid by a rotation matrix and translation vector

        Args:
            rotmat: np.ndarray
            tvec: np.ndarray

        Returns:
            SimpleCuboid
        """
        x, y, z = rotmat @ np.array([self.x, self.y, self.z]) + tvec
        rx, ry, rz = Rotation.from_matrix(rotmat @ self.rotmat).as_euler("xyz")
        return type(self)(
            **self.dict(exclude=["x", "y", "z", "rx", "ry", "rz"]),
            x=x,
            y=y,
            z=z,
            rx=rx,
            ry=ry,
            rz=rz,
        )

    @property
    def posemat(self):
        """Get the pose matrix of the cuboid

        Returns:
            np.ndarray: 4x4 pose matrix
        """
        return np.vstack([np.vstack([self.rotmat.T, np.array([self.x, self.y, self.z])]).T, np.array([0, 0, 0, 1])])

    @property
    def rotmat(self):
        """Get the rotation matrix of the cuboid

        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        # gives more consistent results compared to scipy, same function is used in CVAT framework during labelling
        return geometry.get_rotation_matrix_from_xyz([self.rx, self.ry, self.rz])

    @property
    def tvec(self):
        """Get the translation vector of the cuboid

        Returns:
            np.ndarray: 3x1 translation vector
        """
        return np.array([self.x, self.y, self.z])

    def apply_to_pointcloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Apply the cuboid transformation to a point cloud

        Args:
            pcd: o3d.geometry.PointCloud

        Returns:
            o3d.geometry.PointCloud only corresponding to points within cuboid
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        transform = self.posemat
        transform_inv = np.linalg.pinv(transform)
        points = points @ transform_inv[:3, :3].T + transform_inv[:3, 3]
        filter_x = np.logical_and(points[:, 0] < self.length / 2, points[:, 0] > -self.length / 2)
        filter_y = np.logical_and(points[:, 1] < self.width / 2, points[:, 1] > -self.width / 2)
        filter_z = np.logical_and(points[:, 2] > 0, points[:, 2] < self.height)
        filter = filter_x & filter_y & filter_z
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points[filter, :])
        new_pcd.colors = o3d.utility.Vector3dVector(colors[filter, :])
        return new_pcd

    def interpolate(self, other_box, alpha: float):
        """Interpolate between two cuboids

        Args:
            other_box: SimpleCuboid
            alpha: float

        Returns:
            SimpleCuboid
        """
        assert isinstance(other_box, type(self))
        return type(self)(
            **self.dict(exclude=["x", "y", "z", "rx", "ry", "rz", "length", "width", "height"]),
            length=self.length * (1 - alpha) + other_box.length * alpha,
            width=self.width * (1 - alpha) + other_box.width * alpha,
            height=self.height * (1 - alpha) + other_box.height * alpha,
            x=self.x * (1 - alpha) + other_box.x * alpha,
            y=self.y * (1 - alpha) + other_box.y * alpha,
            z=self.z * (1 - alpha) + other_box.z * alpha,
            rx=self.rx,
            ry=self.ry,
            rz=self.rz,
        )

    @classmethod
    def from_shape(cls, shape, attributes):
        """Create a SimpleCuboid from a shape and attributes

        Args:
            shape: dict
            attributes: dict

        Returns:
            SimpleCuboid
        """
        x, y, z = shape["points"][0], shape["points"][1], shape["points"][2]
        rx, ry, rz = shape["points"][3], shape["points"][4], shape["points"][5] + np.pi / 2
        width, length, height = shape["points"][6], shape["points"][7], shape["points"][8]
        cuboid_id = shape["id"]
        return SimpleCuboid(
            x=x,
            y=y,
            z=(z - height / 2),
            rx=rx,
            ry=ry,
            rz=rz,
            length=length,
            width=width,
            height=height,
            cuboid_id=cuboid_id,
        )

    @classmethod
    def from_annotations(cls, annotation, attributes):
        """Create a map of frame_id -> SimpleCuboid from an annotation

        Args:
            annotation: dict - annotations from CVAT
            attributes: dict - attributes from CVAT

        Returns:
            dict: frame_id -> SimpleCuboid
        """
        frame_cuboids_map = dict()
        track_id = annotation["id"]
        for shape in annotation["shapes"]:
            frame_id = shape["frame"]
            if shape["outside"]:
                continue
            if shape["occluded"]:
                logger.warning("Occluded shape")
                continue
            cuboid = cls.from_shape(shape, attributes)
            cuboid.track_id = track_id
            frame_cuboids_map[frame_id] = cuboid
        return frame_cuboids_map
    

def o3d_bbox_from_cuboid(cuboid: SimpleCuboid):
    """Create an Open3D oriented bounding box from a SimpleCuboid

    Args:
        cuboid: SimpleCuboid

    Returns:
        o3d.geometry.OrientedBoundingBox
    """
    bb_center = np.array([cuboid.x, cuboid.y, cuboid.z])
    bb_center[2] += cuboid.height / 2  # bottom centered cuboid

    return o3d.geometry.OrientedBoundingBox(
        center=bb_center,
        R=cuboid.rotmat,
        extent=np.array([cuboid.length, cuboid.width, cuboid.height]),
    )


def gen_simple_cuboid_from_bbox(bbox: list[float]) -> SimpleCuboid:
    """Generate a SimpleCuboid from a bounding box.
    Args:
        bbox: list[float]
    Returns:
        SimpleCuboid
    """
    x, y, z, length, width, height, rz = bbox
    data = {
        "x": x,
        "y": y,
        "z": z,
        "length": length,
        "width": width,
        "height": height,
        "rz": rz,
        "rx": 0.0,
        "ry": 0.0,
        "cuboid_id": 0,
    }
    return SimpleCuboid(**data)