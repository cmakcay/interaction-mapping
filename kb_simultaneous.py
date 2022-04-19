import numpy as np
import sys
import termios
import tty
import os
import collections
import torch
from PIL import Image as PilImage
from PIL import ImageDraw
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms as transforms
import csv
from scipy.spatial.transform import Rotation as R

from envs import utils
from envs.thor import ThorEnv
from envs.config.config import config_parser

# ROS imports
import rospy
from sensor_msgs.msg import Image
import tf
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

def get_term_character():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def add_rectangle(tensor, bbox):
    img = transforms.ToPILImage()(tensor)
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox,  outline='blue', width=3)
    tensor = transforms.ToTensor()(img)
    return tensor


class ROSHandle:
    def __init__(self):
        self.global_frame_name = "world"
        self.sensor_frame_name = "depth_cam"

        self.color_pub = rospy.Publisher("~color_image", Image, queue_size=100)
        self.depth_pub = rospy.Publisher("~depth_image", Image, queue_size=100)
        self.id_pub = rospy.Publisher("~segmentation_image", Image, queue_size=100)
        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=100)
        self.tf_broadcaster = tf.TransformBroadcaster()
        
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


class KBController():
    def __init__(self, nh):
        # get args
        parser = config_parser()
        self.args = parser.parse_args()

        self.command_dict = {
            '\x1b[A': 'forward',
            'w': 'up',
            's': 'down',
            'a': 'left',
            'd': 'right',
            'e': 'take',
            'r': 'put',
            't': 'misc',
        }

        self.args.num_processes = 1

        self.env = ThorEnv(mode='debug', seed=73745)
        self.obs = self.env.reset()[0]

        self.act_to_idx = collections.defaultdict(lambda: -1)
        self.act_to_idx.update({act:idx for idx, act in enumerate(self.env.get_actions())})

        self.time = 0

        self.N = 5
        self.center = ((self.args.frame_size//self.N)*(self.N//2), (self.args.frame_size//self.N)*(self.N+1)//2)
        self.center_box = [self.center[0], self.center[0], self.center[1], self.center[1]]

        # self.timestamp = open(f"{self.args.save_path}/timestamps.csv", 'w')
        # self.writer = csv.writer(self.timestamp)
        # self.writer.writerow(['ImageID', 'TimeStamp'])

        self.rgbs_to_id = {}
        with open(self.args.csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            # header = next(reader)
            for row in reader:
                rgb_curr  = (int(row["R"]), int(row["G"]), int(row["B"]))
                id_curr = int(row["InstanceID"])
                self.rgbs_to_id[rgb_curr] = id_curr

        # initialize ROS class
        self.nh = nh

        self.render()

        print('KB controller set up.')
        print('↑: move forward, look: wsad, take: e, put: r, misc: t')


    def next_interact_command(self):
        current_buffer = ''
        while True:
            commands = self.command_dict
            current_buffer += get_term_character()
            if current_buffer == 'q' or current_buffer == '\x03':
                break

            if current_buffer in commands:
                yield commands[current_buffer]
                current_buffer = ''
            else:
                match = False
                for k,v in commands.items():
                    if k.startswith(current_buffer):
                        match = True
                        break

                if not match:
                    current_buffer = ''

    def render(self):
        event = self.env.state
        
        # update ros now
        self.nh.update_now()

        # data for panoptic mapping

        # pose of robot camera
        # rotation
        pitch = -event.metadata['agent']['cameraHorizon']        
        yaw = event.metadata['agent']['rotation']['y']
        roll  = event.metadata['agent']['rotation']['z']        
        rotmax = R.from_euler("YXZ",[yaw, pitch, roll], degrees=True)
        rotmax = rotmax.as_matrix()
        
        # translation
        transx = event.metadata['agent']['position']['x']
        transy = event.metadata['agent']['position']['y']
        transz = event.metadata['agent']['position']['z']
        transmat = np.array([[transx], [transy], [transz]])
        
        # rot + trans matrix
        transformat = np.hstack((rotmax, transmat))
        transformat = np.vstack((transformat, [0, 0, 0, 1]))
        self.nh.publish_pose(transformat)
        
        # color + depth + segmentation images
        color_frame = event.frame 
        depth_frame = event.depth_frame
        segmentation_frame = event.instance_segmentation_frame

        if (color_frame is not None):
            im = (PilImage.fromarray(color_frame)).convert("RGB")
            im = np.array(im)
            im_cv = im[:, :, ::-1].copy()
            self.nh.publish_color(im_cv)

        if (depth_frame is not None):
            im = PilImage.fromarray(depth_frame)
            self.nh.publish_depth(im)

        if (segmentation_frame is not None):
            
            id_frame = np.zeros_like(segmentation_frame)
            seg_height = segmentation_frame.shape[0]
            seg_width = segmentation_frame.shape[1]

            # iterate through all pixels to map segmentation rgb to id
            for j_idx in range(seg_width):
              for i_idx in range(seg_height):
                cur_rgb = [segmentation_frame[i_idx,j_idx, :]]  
                cur_rgb_tuple = [tuple(e) for e in cur_rgb]
                cur_id = self.rgbs_to_id[cur_rgb_tuple[0]]
                id_frame[i_idx,j_idx, :] = [cur_id, cur_id, cur_id]
            im = (PilImage.fromarray(id_frame)).convert("RGB")
            im = np.array(im)
            im_cv = im[:, :, ::-1].copy()
            self.nh.publish_id(im_cv)

        frame = torch.from_numpy(np.array(event.frame)).float().permute(2, 0, 1)/255
        frame = F.interpolate(frame.unsqueeze(0), 80, mode='bilinear', align_corners=True)[0]
        frame = add_rectangle(frame, self.center_box)

        utils.show_wait(frame, T=1, win='frame')


    def step(self):
        for action in self.next_interact_command():
            if action == 'misc':
                prompt = ['Misc: done, reset']
                prompt += ['>> ']
                action = input('\n'.join(prompt))
            yield action

    
    def run(self):
        for action in self.step():

            # handle special controller actions
            if action=='done':
                sys.exit(0)

            if action=='reset':
                self.obs = self.env.reset()
                continue

            act_idx = self.act_to_idx[action]

            if act_idx==-1:
                print ('Action not recognized')
                continue

            # handle environment actions
            outputs = self.env.step(act_idx)
            self.obs, reward, done, info = [list(x)[0] for x in zip(outputs)]
            print (f"A: {info['action']} | S: {info['success']} | R: {info['reward']}")

            display = os.environ['DISPLAY']
            os.environ['DISPLAY'] = os.environ['LDISPLAY']
            if action != 'reset':
                self.time += 1
            else:
                self.time = 0
            
            self.render()
            os.environ['DISPLAY'] = display




if __name__ == '__main__':
    os.environ['LDISPLAY'] = os.environ['DISPLAY']

    rospy.init_node('kb_agent')
    kb_agent = ROSHandle()

    controller = KBController(nh=kb_agent)
    controller.run()
