'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import numpy as np
from argparse import ArgumentParser
import os
import sys
from pathlib import Path
import json
import math

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[2]
sys.path.append(dir_path.__str__())

def find_closest_point(p1, d1, p2, d2):
    # Calculate the direction vectors of the lines
    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)

    # Create the coefficient matrix A and the constant vector b
    A = np.vstack((d1_norm, -d2_norm)).T
    b = p2 - p1

    # Solve the linear system to find the parameters t1 and t2
    t1, t2 = np.linalg.lstsq(A, b, rcond=None)[0]

    # Calculate the closest point on each line
    closest_point1 = p1 + d1_norm * t1
    closest_point2 = p2 + d2_norm * t2

    # Calculate the average of the two closest points
    closest_point = 0.5 * (closest_point1 + closest_point2)

    return closest_point


def bound_by_pose(images):
    poses = []
    for w2c in images.values(): # ob_in_cam poses = w2c
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)

    center = np.array([0.0, 0.0, 0.0])
    for f in poses:
        src_frame = f[0:3, :]
        for g in poses:
            tgt_frame = g[0:3, :]
            p = find_closest_point(src_frame[:, 3], src_frame[:, 2], tgt_frame[:, 3], tgt_frame[:, 2])
            center += p
    center /= len(poses) ** 2

    radius = 0.0
    for f in poses:
        radius += np.linalg.norm(f[0:3, 3])
    radius /= len(poses)
    bounding_box = [
        [center[0] - radius, center[0] + radius],
        [center[1] - radius, center[1] + radius],
        [center[2] - radius, center[2] + radius],
    ]
    return center, radius, bounding_box

def _cv_to_gl(cv):
    # convert to GL convention used in iNGP
    gl = cv * np.array([1, -1, -1, 1])
    return gl

def transform_pose_opengl_to_opencv(pose):
    # Inverse transform is identical
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    return pose @ flip_yz

def export_to_json(K, w, h, transforms, bounding_box, center, radius, file_path):
    fl_x = K[0, 0]
    fl_y = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2

    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "sk_x": 0.0,  # TODO: check if colmap has skew
        "sk_y": 0.0,
        "k1": 0.0,  # take undistorted images only
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "is_fisheye": False,  # TODO: not supporting fish eye camera
        "cx": cx,
        "cy": cy,
        "w": int(w),
        "h": int(h),
        "aabb_scale": np.exp2(np.rint(np.log2(radius))),  # power of two, for INGP resolution computation
        "aabb_range": bounding_box,
        "sphere_center": center,
        "sphere_radius": radius,
        "frames": [],
    }

    # read poses
    for transform_name in sorted(transforms.keys()):
        w2c = transforms[transform_name]
        c2w = np.linalg.inv(w2c)
        c2w = transform_pose_opengl_to_opencv(c2w)  # convert to GL convention used in iNGP

        frame = {"file_path": "rgb/" + transform_name.split(".")[0] + ".png", "mask_path": "masks/" + transform_name.split(".")[0] + ".png", "transform_matrix": c2w.tolist()}
        out["frames"].append(frame)

    with open(file_path, "w") as outputfile:
        json.dump(out, outputfile, indent=2)

    return

def read_transforms(data_dir):
    transforms = {}
    for image_name in sorted(os.listdir(os.path.join(args.data_dir, "rgb"))):
        transform_name = image_name.split(".")[0] + ".txt"
        T = np.loadtxt(os.path.join(os.path.join(args.data_dir, "ob_in_cam"), transform_name))
        transforms[transform_name] = T
    
    return transforms

def data_to_json(args):
    transforms = read_transforms(args.data_dir)
    K = np.loadtxt(os.path.join(args.data_dir, "cam_K.txt"))

    # Only handle object scene type
    center, radius, bounding_box = bound_by_pose(transforms)

    # export json file
    export_to_json(K, args.width, args.height, transforms, bounding_box, list(center), radius, os.path.join(args.data_dir, "transforms.json"))
    print("Writing data to json file: ", os.path.join(args.data_dir, "transforms.json"))
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data")
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width of images",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of images",
    )
    args = parser.parse_args()
    data_to_json(args)
