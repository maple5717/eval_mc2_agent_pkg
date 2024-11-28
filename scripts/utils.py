import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
# import ros_numpy
from std_msgs.msg import Header

def numpy_to_occupancy_grid(numpy_array, frame_id="base_link", resolution=0.05):
    """
    Convert a binary NumPy array to a ROS OccupancyGrid message and center the map at the origin.

    Args:
        numpy_array (np.ndarray): Binary NumPy array (0 and 1).
        frame_id (str): The frame_id for the OccupancyGrid message.
        resolution (float): The resolution of the grid (meters per cell).

    Returns:
        OccupancyGrid: The ROS OccupancyGrid message.
    """
    # Convert binary values to occupancy values (0: free, 100: occupied)
    occupancy_values = (numpy_array * 100).astype(np.uint8).flatten()
    
    # Calculate origin to center the map
    height, width = numpy_array.shape
    origin_x = -width * resolution / 2.0
    origin_y = -height * resolution / 2.0
    
    # Create OccupancyGrid message
    msg = OccupancyGrid()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.info.resolution = resolution
    msg.info.width = width
    msg.info.height = height
    msg.info.origin.position.x = origin_x
    msg.info.origin.position.y = origin_y
    msg.info.origin.position.z = 0
    msg.info.origin.orientation.x = 0
    msg.info.origin.orientation.y = 0
    msg.info.origin.orientation.z = 0
    msg.info.origin.orientation.w = 1
    msg.data = occupancy_values.tolist()
    
    return msg