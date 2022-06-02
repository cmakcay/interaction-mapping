import numpy as np

# ROS imports
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import tf
from geometry_msgs.msg import PoseStamped


class Bridge:
    def __init__(self, env_index):
        self.global_frame_name = "world"
        self.sensor_frame_name = "depth_cam"
        self.reward = 0
        self.env_index = env_index
        self.evaluation = 0

        self.color_pub = rospy.Publisher(f"~color_image_{env_index}", Image, queue_size=100)
        self.depth_pub = rospy.Publisher(f"~depth_image_{env_index}", Image, queue_size=100)
        self.id_pub = rospy.Publisher(f"~segmentation_image_{env_index}", Image, queue_size=100)
        self.pose_pub = rospy.Publisher(f"~pose_{env_index}", PoseStamped, queue_size=100)
        self.tf_broadcaster = tf.TransformBroadcaster()
        # self.reward_sub = rospy.Subscriber(f"/panoptic_mapper_{env_index}/reward", Int8, callback=self.reward_callback, queue_size=100)
        self.time_pub = rospy.Publisher(f"time_{env_index}", Int16, queue_size=100)

        self.cv_bridge = CvBridge()
        self.now = rospy.Time.now()

    def update_now(self):
        self.now = rospy.Time.now()

    def publish_pose(self, pose):
        pose = pose.flatten()
        pose_data = [float("{:.6f}".format(x)) for x in pose]
        transform = np.eye(4)
        for row in range(4):
            for col in range(4):
                transform[row, col] = pose_data[row * 4 + col]
        rotation = tf.transformations.quaternion_from_matrix(transform)
        self.tf_broadcaster.sendTransform(
            (transform[0, 3], transform[1, 3], transform[2, 3]), rotation,
            self.now, self.sensor_frame_name, self.global_frame_name)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.now
        pose_msg.header.frame_id = self.global_frame_name
        pose_msg.pose.position.x = pose_data[3]
        pose_msg.pose.position.y = pose_data[7]
        pose_msg.pose.position.z = pose_data[11]
        pose_msg.pose.orientation.x = rotation[0]
        pose_msg.pose.orientation.y = rotation[1]
        pose_msg.pose.orientation.z = rotation[2]
        pose_msg.pose.orientation.w = rotation[3]
        self.pose_pub.publish(pose_msg)

    def publish_color(self, cv_img):
        img_msg = self.cv_bridge.cv2_to_imgmsg(cv_img, "bgr8")
        img_msg.header.stamp = self.now
        img_msg.header.frame_id = self.sensor_frame_name
        self.color_pub.publish(img_msg)
    
    def publish_depth(self, cv_img):
        img_msg = self.cv_bridge.cv2_to_imgmsg(np.array(cv_img), "32FC1")
        img_msg.header.stamp = self.now
        img_msg.header.frame_id = self.sensor_frame_name
        self.depth_pub.publish(img_msg)

    def publish_id(self, cv_img):
        img_msg = self.cv_bridge.cv2_to_imgmsg(cv_img[:, :, 0], "8UC1")
        img_msg.header.stamp = self.now
        img_msg.header.frame_id = self.sensor_frame_name
        self.id_pub.publish(img_msg)

    # def reward_callback(self, data):
    #     print(f"reward update = {rospy.Time.now()}")
    #     self.reward = data.data

    def get_reward(self):
        try:
            data = rospy.wait_for_message(f"/panoptic_mapper_{self.env_index}/reward", Float64, timeout=0.5)
            self.reward = int(data.data)
        except rospy.ROSException:
            self.reward = 0
        return self.reward

    def get_evaluation(self):
        try:
            data = rospy.wait_for_message(f"/panoptic_mapper_{self.env_index}/evaluation", Float64, timeout=0.5)
            self.evaluation = int(data.data)
        except rospy.ROSException:
            self.evaluation = -1
        return self.evaluation

    def publish_time(self, t):
        time_msg = Int16()
        time_msg.data = t
        self.time_pub.publish(time_msg)