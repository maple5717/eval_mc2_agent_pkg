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
import tf
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge, CvBridgeError
import numpy as np 
import tf.transformations as tft
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist
import keyboard

def quaternion_to_rpy(q):
    if isinstance(q, list):
        x, y, z, w = q
    else:
        x, y, z, w = q.x, q.y, q.z, q.w

    # Convert quaternion to roll, pitch, yaw (RPY)
    roll, pitch, yaw = tft.euler_from_quaternion([x, y, z, w])
    return roll, pitch, yaw

# 2>&1 | grep -Ev "TF_REPEATED_DATA|278"
class Evaluator:
    def __init__(self, agent, action_freq=1/10):
        # rospy.set_param('/use_sim_time', True)
        rospy.init_node('mc2_agent_evaluator', anonymous=True)
        self.camera_height = 1.394
        self.agent = agent  
        self.obs = self.__load_init_obs()
        self.start = False
        self.last_received_time = rospy.Time.now()
        self.use_agent = True
        
        self.bridge = CvBridge()

        # subscribers
        rgb_sub = Subscriber('/camera/camera/color/image_raw', Image)
        depth_sub = Subscriber('/camera/camera/aligned_depth_to_color/image_raw', Image)
        
        self.tf_listener = tf.TransformListener()
        # rospy.Timer(rospy.Duration(1 / action_freq), self.__agent_callback)
        self.subb = rospy.Subscriber('/cmd_vel', Twist, self.__agent_callback)
        rospy.Timer(rospy.Duration(1), self.__watchdog_callback)
        rospy.Timer(rospy.Duration(0.01), self.__tf_callback) 

        self.source_frame = 'odom'
        self.target_frame = 'base_link'

        # Synchronizer with a queue size of 10 and 0.1s time tolerance
        self.ats = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=100, slop=0.05)
        self.ats.registerCallback(self.__rgbd_callback)



    def run(self):
        if self.use_agent:
            self.agent.reset()
        self.last_received_time = rospy.Time.now()
        rospy.loginfo("Evaluator is ready to receive commands! ")
        rospy.spin()

    def __watchdog_callback(self, event):
        if (rospy.Time.now() - self.last_received_time).to_sec() > 1:
            self.start = 0
            
    def __agent_callback(self, event):
        if self.start:
            if self.use_agent:
                self.agent.act(self.obs)

    def __tf_callback(self, event):    
        # Get the transformation between the frames
        try:
            (trans, q) = self.tf_listener.lookupTransform(self.source_frame, self.target_frame, rospy.Time(0))

            theta = quaternion_to_rpy(q)[2]
            self.obs.gps = np.array(trans)
            self.obs.compass = np.array([theta])

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr("Error while looking up transform: %s", str(e))

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

        obs.xyz = np.ones([640, 480])
        return obs
    
    def __get_cam_pose(self):
        x, y = self.obs.gps[0], self.obs.gps[1]
        theta = self.obs.compass[0]
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

        cam_pose = robot_pose @ robot2cam
        return cam_pose

    def __obs_generation(self, rgb_img, depth_img): 
        self.obs.rgb = rgb_img.astype(np.uint8)
        self.obs.depth = depth_img.astype(np.float32)

        self.obs.camera_pose = self.__get_cam_pose()
        
        # self.agent.act(self.obs)

    def __rgbd_callback(self, rgb_msg, depth_msg):
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
        # if "y" == input("continue?")
        #     agent.run()

        self.__obs_generation(rgb_img, depth_img)
        # print(self.trans)
        self.start = True
        self.last_received_time = rospy.Time.now()
        
        
    