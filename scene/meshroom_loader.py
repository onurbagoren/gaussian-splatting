#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import collections
import struct


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params", "name"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_extrinsics(poses_dict):
    images = {}
    L = len(poses_dict)
    for ii in range(L):
        image_id = int(poses_dict[ii]["poseId"])
        rotmat = poses_dict[ii]["pose"]["transform"]["rotation"]
        tvec = np.array(poses_dict[ii]["pose"]["transform"]["center"]).astype(
            np.float64
        )
        qvec = rotmat2qvec(np.array(rotmat).reshape(3, 3).astype(np.float64))
        images[image_id] = BaseImage(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=image_id,
            name="",
            xys=None,
            point3D_ids=None,
        )

    assert len(images) == L
    return images


def read_params(intrinsics_dict):
    # get the camera intrinsics
    width = float(intrinsics_dict["width"])
    height = float(intrinsics_dict["height"])
    sensor_width = float(intrinsics_dict["sensorWidth"])
    sensor_height = float(intrinsics_dict["sensorHeight"])
    focal_length = float(intrinsics_dict["focalLength"])
    principal_point = intrinsics_dict["principalPoint"]
    distortion_params = intrinsics_dict["distortionParams"]
    dist = [float(p) for p in distortion_params]
    k1 = dist[0]
    k2 = dist[1]
    k3 = dist[2]
    dist = np.array([k1, k2, 0, 0, k3])  # distortion parameters

    f_x = (focal_length * width) / sensor_width
    f_y = (focal_length * height) / sensor_height
    c_x = (width / 2) + float(principal_point[1])
    c_y = (height / 2) - float(principal_point[0])

    return f_x, f_y, c_x, c_y


def read_intrinsics(views_dict, intrinsics_dict):
    cameras = {}
    L = len(views_dict)
    for ii in range(L):
        camera_id = int(views_dict[ii]["viewId"])
        model_id = 1
        model_name = CAMERA_MODEL_IDS[model_id].model_name
        num_params = CAMERA_MODEL_IDS[model_id].num_params
        width = float(views_dict[ii]["width"])
        height = float(views_dict[ii]["height"])
        name = views_dict[ii]["frameId"]

        
        """
        The parameters are from the ideal camera model.
        Meshroom output is a radial3 camera type,
        so we point the image directory to the undistorted
        images and use these intrinsics
        """
        params = read_params(intrinsics_dict[0])
        cameras[camera_id] = Camera(
            id=camera_id,
            model=model_name,
            width=width,
            height=height,
            params=np.array(params),
            name=f"{int(name):05d}.jpeg",
        )
    assert len(cameras) == L
    return cameras
