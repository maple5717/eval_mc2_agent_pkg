#! /usr/bin/env python
from home_robot.core.interfaces import DiscreteNavigationAction, Observations 

import argparse
import os
# from mc2.agent.mc2_agent.mc2_agent import MC2Agent

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from cv_bridge import CvBridge, CvBridgeError
import rospy
# import tf
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge, CvBridgeError
import numpy as np 
import tf.transformations as tft
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Twist
# from pointcloud_utils import pointcloud2_to_xyz_feats
import pickle 
import copy 
import cv2
from utils import *

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

def save_observations(observations, filename):
    with open(filename, 'wb') as file:
        pickle.dump(observations, file)

def quaternion_to_rpy(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    # Convert quaternion to roll, pitch, yaw (RPY)
    roll, pitch, yaw = tft.euler_from_quaternion([x, y, z, w])
    return roll, pitch, yaw


class Evaluator:
    def __init__(self, agent, action_freq=1/2):
        rospy.init_node('mc2_agent_evaluator', anonymous=True)
        self.camera_height = 1.394
        self.agent = agent  
        self.obs = self.__load_init_obs()
        self.start = False
        self.last_received_time = rospy.Time.now()
        
        self.bridge = CvBridge()
        self.data_arr = []

        # subscribers
        # odom_sub = Subscriber('/vins_estimator/odometry', Odometry)
        odom_sub = Subscriber('/transformed_odom', Odometry)
        # odom_sub = Subscriber('/poseupdate', PoseWithCovarianceStamped)
        # self.tf_listener = tf.TransformListener()
        rospy.Timer(rospy.Duration(1 / action_freq), self.__agent_callback)
        # self.subb = rospy.Subscriber('/cmd_vel', Twist, self.__agent_callback)
        rospy.Timer(rospy.Duration(1), self.__watchdog_callback)

        # Synchronizer with a queue size of 10 and 0.1s time tolerance
        rgb_sub = Subscriber('/camera/camera/color/image_raw', Image)
        depth_sub = Subscriber('/camera/camera/aligned_depth_to_color/image_raw', Image)
        if True:
            self.ats = ApproximateTimeSynchronizer([rgb_sub, depth_sub, odom_sub], queue_size=100, slop=0.05)
            self.ats.registerCallback(self.__rgbd_callback)
        else:
            pc_sub = Subscriber('/camera/camera/depth/color/points', PointCloud2)
            self.ats = ApproximateTimeSynchronizer([rgb_sub, depth_sub, pc_sub, odom_sub], queue_size=100, slop=0.05)
            self.ats.registerCallback(self.__pointcloud_callback)

        self.img_vis = np.zeros([480,480])
        self.map_pub = rospy.Publisher('occupancy_grid', OccupancyGrid, queue_size=10)

    def run(self):
        self.agent.reset()
        self.last_received_time = rospy.Time.now()
        rospy.loginfo("Evaluator is ready to receive commands! ")
        rospy.spin()

    def __watchdog_callback(self, event):
        if (rospy.Time.now() - self.last_received_time).to_sec() > 20:
            self.start = 0

        # print(f"{self.img_vis.sum()}\n"*10)
        # cv2.imshow('Map Image Viewer', np.ones([480, 480, 3], dtype=np.uint8)*255 ) #self.img_vis * 255
        # rospy.loginfo(self.img_vis.max())
        # cv2.waitKey(0.45)
            
    def __agent_callback(self, event):
        if self.start:
            # cv2.imshow('Image Window', self.obs.rgb)
            # cv2.waitKey(0)
            action, info = self.agent.act(self.obs)
            map_img = info['obstacle_map']  # goal_map, frontier_map, explored_map


            grid_msg = numpy_to_occupancy_grid(map_img.astype(np.uint8))
            self.map_pub.publish(grid_msg)
            
        else:
            rospy.logwarn("Agent not initialized. Did you publish data to the agent? ")
            # aaa = False

    def __load_init_obs(self):
        obs = Observations(gps=[0,0], compass=[0], rgb=[0], depth=[0]) 
        obs.camera_K = np.array([[605.2938, 0, 316.6696], 
                                 [0, 605.4811, 236.8994], 
                                 [0, 0, 1]])
        # np.load("d435i_intrinsics.npy")
        
        task_obs = {"goal_name": 'Move knife from cabinet to table', 
                    'object_name': "knife",
                    "start_recep_name": "cabinet"} # generate fake task obs
        obs.task_observations = task_obs

        obs.xyz = None # np.ones([640, 480])
        return obs
    
    def __get_cam_pose(self, x, y, theta):
        robot_pose = np.eye(4)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        rotation_matrix_so2 = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])

        robot_pose[:2, :2] = rotation_matrix_so2
        robot_pose[:-1, 3] = [x, y, self.camera_height]

        robot2cam = np.array([[0,-1,0,0],
                              [0,0,-1,0],
                              [1,0,0,0],
                              [0,0,0,1]], dtype=np.float32).T
        # print("____________")
        # print(robot_pose)
        # print(robot2cam)
        # print(robot_pose @ robot2cam)
        cam_pose = robot_pose @ robot2cam
        return cam_pose

    def __obs_generation_from_img(self, rgb_img, depth_img, pose): 
        self.obs.rgb = rgb_img.astype(np.uint8)
        self.obs.depth = depth_img.astype(np.float32)

        # print(self.obs.rgb.shape, self.obs.depth.shape)

        pos = pose.position
        orn_q = pose.orientation 
        theta = quaternion_to_rpy(orn_q)[2]

        self.obs.gps = np.array([pos.x, pos.y])
        self.obs.compass = np.array([theta])

        self.obs.camera_pose = self.__get_cam_pose(pos.x, pos.y, theta)

    def __obs_generation_from_pointcloud(self, rgb_img, depth_img, xyz, feat, pose): 
        self.obs.xyz = xyz
        self.obs.rgb = rgb_img # rgb_img.astype(np.uint8)
        self.obs.depth = depth_img
        # self.obs.depth = None
        # self.obs.feats = feat

        pos = pose.position
        orn_q = pose.orientation 
        theta = quaternion_to_rpy(orn_q)[2]

        self.obs.gps = np.array([pos.x, pos.y])
        self.obs.compass = np.array([theta])

        self.obs.camera_pose = self.__get_cam_pose(pos.x, pos.y, theta + np.pi/2)

        self.data_arr.append(copy.deepcopy(self.obs))
        save_observations(self.data_arr, "saved_obs.pkl")
        rospy.loginfo("saved")
        
        # self.agent.act(self.obs)

    def __rgbd_callback(self, rgb_msg, depth_msg, odom):
        """
            Generate the observation instance given RGBD
        """

        try: 
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        depth_img = depth_img.astype(np.float32) * 0.001
        # print(depth_img.min(), depth_img.max())
        self.__obs_generation_from_img(rgb_img, depth_img, odom.pose.pose)
        
        # delay_ms = (rgb_msg.header.stamp.to_sec()-odom.header.stamp.to_sec())*1000
        # if delay_ms > 30:
        #     print(delay_ms)
        # if "y" == input("continue?")
        #     agent.run()
        self.start = True
        # self.last_received_time = rospy.Time.now()
        # msg_delay = (rgb_msg.header.stamp - self.last_received_time).to_sec()

        # rospy.logerr(msg_delay)

    def __pointcloud_callback(self, rgb_msg, depth_msg, pointcloud_msg, odom):
        """
            Generate the observation instance given RGBD
        """
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth_img = depth_img.astype(np.float32) * 0.001

        xyz, feat = pointcloud2_to_xyz_feats(pointcloud_msg)

        # print(depth_img.min(), depth_img.max())
        self.__obs_generation_from_pointcloud(rgb_img, depth_img, xyz, feat, odom.pose.pose)
        
        # delay_ms = (pointcloud_msg.header.stamp.to_sec()-odom.header.stamp.to_sec())*1000
        # if delay_ms > 30 * 0:
        #     print(delay_ms)
        # if "y" == input("continue?")
        #     agent.run()
        self.start = True
        self.last_received_time = rospy.Time.now()
        
        
    