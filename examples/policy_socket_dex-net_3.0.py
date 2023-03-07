# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Displays robust grasps planned using a GQ-CNN grapsing policy on a set of saved
RGB-D images. The default configuration for the standard GQ-CNN policy is
`cfg/examples/cfg/examples/gqcnn_pj.yaml`. The default configuration for the
Fully-Convolutional GQ-CNN policy is `cfg/examples/fc_gqcnn_pj.yaml`.

Author
------
Jeff Mahler & Vishal Satish
"""
import argparse

import json
import os
import time

import numpy as np

from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage, RgbdImage)
from visualization import Visualizer2D as vis

from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)

import sys
print(sys.path)

from gqcnn.utils import GripperMode

import socket
import open3d as o3d
import cv2
from camera import IntelCamera
import copy


def get_suction_point_3d(depth_image: np, suction_point: np, cam_instance):
    global pcd

    xyz = cam_instance.generate(depth_image)
    # pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
        
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    try:
        pcd.orient_normals_towards_camera_location()
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        suction_point = np.reshape(suction_point, (2, ))
    except:
        return 0
        
    ## creat xyz for the final suction point
    if cam_instance.device_product_line == "D400" or cam_instance.device_product_line == "AzureKinect":
        Z = depth_image[suction_point[0]][suction_point[1]]*0.001
    elif cam_instance.device_product_line == "L500":
        Z = depth_image[suction_point[0]][suction_point[1]]*0.00025
    X = (suction_point[1]-cam_instance.camera_mat[0][2])*Z/cam_instance.camera_mat[0][0]
    Y = (suction_point[0]-cam_instance.camera_mat[1][2])*Z/cam_instance.camera_mat[1][1]

    ## select most similar point among the generated points
    target = np.array([X, Y, Z])
    target = np.reshape(target, (1, 3))
    index = np.linalg.norm((xyz-target), axis=1)
    index = np.argmin(index)
    point = np.asarray(pcd.points)[index]
    normal = np.asarray(pcd.normals)[index]

    return point, normal


def get_seg_mask(depth_image, cam_instance):
    '''
    - prediction map is 2d with single channel
    - could extract pixel index which got some level of probability
    - convert depth map to point cloud only for the selected pixel
    - filter points out of the roi
    - re-project points inside the roi
    - filter the pixel out of the roi using indices from depth map 
    '''

    width = cam_instance.intrinsic_o3d.width
    height = cam_instance.intrinsic_o3d.height
    fxy = cam_instance.intrinsic_o3d.get_focal_length()
    cxy = cam_instance.intrinsic_o3d.get_principal_point()

    # print(np.shape(depth_image))
    # print(np.shape(prediction_mask))

    ## render point cloud
    depth_image = depth_image.astype(np.uint16)

    depth_image = np.reshape(depth_image, (540, 960))

    cam_instance.generate(depth_image, downsample=False)
    xyz = cam_instance.crop_points()
    pcd = cam_instance.pcd
    # o3d.visualization.draw_geometries([pcd])

    roi_mask = np.full((height, width, 1), 0, dtype=np.uint8)
    for p in xyz:
        px = (fxy[0]*p[0] + cxy[0]*p[2])//p[2]
        py = (fxy[1]*p[1] + cxy[1]*p[2])//p[2]
        px = int(px)
        py = int(py)
        roi_mask[py][px] = [255]
    
    roi_mask = cv2.dilate(roi_mask, np.ones((2, 2), np.uint8))
    return roi_mask
    # cv2.imshow("roi_mask", object_candidate[0, :, :]*255)
    # cv2.waitKey()

cam = IntelCamera([])

pcd = o3d.geometry.PointCloud()

# Set up logger.
logger = Logger.get_logger("examples/policy.py")

# set client information
SERVER_IP = '192.168.1.48'
SERVER_PORT = 9999
SIZE = 1024
SERVER_ADDR = (SERVER_IP, SERVER_PORT)

DATASock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

data_connection = DATASock.connect_ex(SERVER_ADDR)

if data_connection == 0:
    print("-"*100)
    print("CONNECTED TO SERVER!!")
    print("-"*100)

else:
    print("-"*100)
    print("**FAILED** TO CONNECT!!")
    print("-"*100)

model_path = '/home/son-skku/gqcnn/models/GQCNN-3.0'

gripper_mode = GripperMode.LEGACY_SUCTION

# Read config.
config = YamlConfig('cfg/examples/replication/dex-net_3.0.yaml')
inpaint_rescale_factor = config["inpaint_rescale_factor"]
policy_config = config["policy"]

# Make relative paths absolute.
if "gqcnn_model" in policy_config["metric"]:
    policy_config["metric"]["gqcnn_model"] = model_path
    if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
        policy_config["metric"]["gqcnn_model"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            policy_config["metric"]["gqcnn_model"])

# Setup sensor.
camera_intr = CameraIntrinsics.load('data/calib/primesense/primesense.intr')

for i in range(10):
    cam.stream()

rgb, depth = cam.stream()

org_depth = copy.deepcopy(depth)

segmask = get_seg_mask(org_depth, cam)
cv2.imwrite('segmask.png', segmask)

# Read images.
# depth_data = np.load(depth)
depth = depth*0.00025
depth_im = DepthImage(depth, frame=camera_intr.frame)
color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                3]).astype(np.uint8),
                        frame=camera_intr.frame)

# Optionally read a segmask.
# segmask = depth_im.invalid_pixel_mask().inverse()

segmask = BinaryImage.open('segmask.png')
valid_px_mask = depth_im.invalid_pixel_mask().inverse()
segmask = segmask.mask_binary(valid_px_mask)

# Inpaint.
depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)

if "input_images" in policy_config["vis"] and policy_config["vis"][
        "input_images"]:
    vis.figure(size=(10, 10))
    num_plot = 1
    if segmask is not None:
        num_plot = 2
    vis.subplot(1, num_plot, 1)
    vis.imshow(depth_im)
    if segmask is not None:
        vis.subplot(1, num_plot, 2)
        vis.imshow(segmask)
    vis.show()

# Create state.
rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

policy = CrossEntropyRobustGraspingPolicy(policy_config)

# Query policy.
policy_start = time.time()
action = policy(state)
logger.info("Planning took %.3f sec" % (time.time() - policy_start))

# Vis final grasp.
if policy_config["vis"]["final_grasp"]:
    vis.figure(size=(10, 10))
    vis.imshow(rgbd_im.depth,
                vmin=policy_config["vis"]["vmin"],
                vmax=policy_config["vis"]["vmax"])
    vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
    vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
        action.grasp.depth, action.q_value))
    
    print("action.grasp.center[0]: ",  action.grasp.center[0])
    print("action.grasp.center[1]: ",  action.grasp.center[1])
    print("action.grasp.depth: ",  action.grasp.depth)

    vis.show()

suction_point_2d = np.array([int(action.grasp.center[1]), int(action.grasp.center[0])])
print("suction_point_2d:", suction_point_2d)

point, normal = get_suction_point_3d(org_depth, suction_point_2d, cam)
print("suction point:", point)

msg = str(point[0])+','+str(point[1])+','+str(point[2])+','+str(normal[0])+','+str(normal[1])+','+str(normal[2])
DATASock.send(msg.encode())
DATASock.close()