'''
The code is adapted from Facebook's interaction-exploration https://github.com/facebookresearch/interaction-exploration.git
'''
import collections
import gym
from gym import spaces
from .config.config import config_parser
import numpy as np
from PIL import Image
import itertools
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

class ThorEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self, mode, seed):
        super(ThorEnv, self).__init__()

        # get args
        parser = config_parser()
        self.args = parser.parse_args()

        # set gym env args
        self.observations = self.args.observations
        platform = CloudRendering if self.args.headless == True else None
        x_display = self.args.x_display
        local_exe = None if self.args.local_exe=='None' else self.config.local_exe
        obs_size = self.args.obs_size 
        
        if self.observations == "rgb":
            num_channels = 3
        elif self.observations == "rgbd":
            num_channels == 4
        elif self.observations == "rgba":
            num_channels = 5
        elif self.observations == "rgbda":
            num_channels = 6
        else:
            raise NotImplementedError

        
        # set global args for the class
        self.rs = np.random.RandomState(seed) #set random seed for same training data
        self.init_params = {
            'gridSize': 0.25,
            'renderDepthImage': True,
            'renderInstanceSegmentation': True,
            'visibilityDistance': 1.0, #maybe switch to 1.5?
            'width': 80, #setting height and width here doesnt work, also set below
            'height': 80, 
        }
        self.mode = mode # train or eval
        self.rot_size_x = self.args.rot_size_x
        self.rot_size_y = self.args.rot_size_y
        self.frame_sz = self.args.frame_size
        self.max_t = self.args.num_steps
        self.reward_type = self.args.reward_type

        self.eval_episodes = self.args.eval_episodes # use predefined scenes and episodes for evaluation
        self.eval_scenes = self.args.eval_scenes
        self.eval_episode_iter = iter(self.eval_episodes)
        self.eval_scene_iter = iter(self.eval_scenes)
        self.debug_scene = self.args.debug_scene # use predefined scene for debugging
        self.debug_episode = self.args.debug_episode
        
        # reward specific parameters
        if self.reward_type == "interaction_count":
            self.interaction_count = collections.defaultdict(int)
        elif self.reward_type == "interaction_navigation":    
            self.interaction_count = collections.defaultdict(int)
            self.camera_poses = collections.defaultdict(int)
        
        # only navigation + take and put
        self.action_type = self.args.action_type
        self.actions = ["forward", "up", "down", "right", "left", "take", "put"]
        self.action_functions = {
            'forward':self.move,
            'up':self.look,
            'down':self.look,
            'right':self.turn,
            'left':self.turn,
            'take':self.take,
            'put':self.put,
        }

        # define action space and observation space
        # ai2thor images are already uint8 (0-255)
        self.observation_space = spaces.Box(low=0, high=255, shape=(num_channels, obs_size, obs_size), dtype=np.uint8)
        if self.action_type == 'with_int':
            self.action_space = spaces.Discrete(len(self.actions))
        elif self.action_type == 'without_int':
            self.action_space = spaces.Discrete(len(self.actions)-2)
        
        # take/put grid
        self.N = 5
        self.center = ((self.frame_sz//self.N)*(self.N//2), (self.frame_sz//self.N)*(self.N+1)//2)
        self.center_grid = np.array([[self.center[0], self.center[0], self.center[1], self.center[1]]]) # xyxy     

        # create ai2thor controller
        self.controller = Controller(quality='Ultra', local_executable_path=local_exe, 
                                x_display=x_display, width=self.frame_sz, height=self.frame_sz, platform=platform)

    # functions need to be overwritten for gym env
    def step(self, action):
        self.t += 1
        self.step_info = self.act(self.actions[action])
        observation = self.get_observation(self.state)
        reward = self.get_reward()
        done = self.get_done()
        self.step_info.update({'reward':reward, 'done':done})
        return observation, reward, done, self.step_info

    def reset(self):
        self.init_env()
        return self.get_observation(self.state)

    def close(self):
        self.controller.stop()

    def render(self):
        pass


    # getters
    def get_reward(self):
        if self.reward_type == "interaction_count":
            reward = 0
            info = self.step_info
            if (info['action'] == "take" or info['action'] == "put") and info['success']:
                key = (info['action'], info['params']['objectId'])
                if key not in self.interaction_count:
                    reward += 1.0
                    self.interaction_count[key] += 1
            return reward        
        elif self.reward_type == "interaction_navigation":
            int_reward = 0
            nav_reward = 0
            info = self.step_info
            if (info['action'] == "take" or info['action'] == "put") and info['success']:
                key = (info['action'], info['params']['objectId'])
                if key not in self.interaction_count:
                    int_reward += 1.0
                    self.interaction_count[key] += 1
 
            x, y, z, rot, hor = self.agent_pose(self.state)
            # pose_key = (x, y, z , rot, hor)
            pose_key = (x, y, z)
            if pose_key not in self.camera_poses:
                nav_reward+=0.2
                self.camera_poses[pose_key]+=1
            reward = int_reward+nav_reward
            return reward        
        else:
            raise NotImplementedError

    def get_done(self):
        return self.t>=self.max_t
    
    # borrowed from Facebook's interaction-exploration
    def get_target_obj(self, obj_property, overlap_thresh=0.3):

        objId_to_obj = {obj['objectId']:obj for obj in self.state.metadata['objects'] if obj['visible'] and obj['objectId']!=self.inv_obj}

        instance_segs = self.state.instance_segmentation_frame # (300, 300, 3)
        color_to_count = Image.fromarray(instance_segs, 'RGB').getcolors()
        color_to_count = dict({pix:cnt for cnt,pix in color_to_count})
        color_to_objId = self.state.color_to_object_id
        active_px = instance_segs[self.center[0]:self.center[1], self.center[0]:self.center[1]] # (B, B, 3)
        S = active_px.shape[0]
        instance_counter = collections.defaultdict(list)
        for i, j in itertools.product(range(S), range(S)):
            color = tuple(active_px[i, j])    
            if color not in color_to_objId or color_to_objId[color] not in objId_to_obj:
                continue            
            instance_counter[color].append(np.abs(i-S//2) + np.abs(j-S//2))
        instance_counter = [{'color':color, 'N':len(scores), 'objectId':color_to_objId[color], 'dist':np.mean(scores), 'p1':len(scores)/S**2, 'p2':len(scores)/color_to_count[color]} for color, scores in instance_counter.items()]

        # either >K% of the object is inside the box, OR K% of the pixels belong to that object
        all_targets = [inst for inst in instance_counter if inst['p1']>overlap_thresh or inst['p2']>overlap_thresh]
        all_targets = sorted(all_targets, key=lambda x: x['dist'])
        act_targets = [candidate for candidate in all_targets if obj_property(objId_to_obj[candidate['objectId']])]
        
        targets = {'objectId':None, 'obj':None, 'center_objectId':None, 'center_obj':None, 'int_target':None}
        if len(all_targets)>0:
            objId = all_targets[0]['objectId']
            targets.update({'center_objectId':objId, 'center_obj':objId_to_obj[objId], 'int_target':'center_obj'})

        if len(act_targets)>0:
            objId = act_targets[0]['objectId']
            targets.update({'objectId':objId, 'obj':objId_to_obj[objId], 'int_target':'obj'})

        if targets['int_target'] is None:
            targets['int_target'] = 'dummy'

        return targets

    def get_actions(self):
        return self.actions

    def get_observation(self, state):
        if self.observations == "rgb":
            img = np.array(state.frame, dtype=np.uint8)
            img_channel_first = np.moveaxis(img, -1, 0)
            obs = img_channel_first
        elif self.observations == "rgbd":
            img = np.array(state.frame, dtype=np.uint8)
            img_channel_first = np.moveaxis(img, -1, 0)
            depth_frame = state.depth_frame[np.newaxis, ...]*50.0
            img_channel_first = np.append(img_channel_first, depth_frame, axis=0)
            obs = img_channel_first
        elif self.observations == "rgba":
            img = np.array(state.frame, dtype=np.uint8)
            img_channel_first = np.moveaxis(img, -1, 0)

            curr_objects = state.metadata['objects']
            num_obj_actions = 2 # [take, put]
            affordance_mask = np.zeros((num_obj_actions, self.args.obs_size, self.args.obs_size), dtype=np.uint8)
            seg_frame = state.instance_segmentation_frame

            color_to_id = state.color_to_object_id
            seg_height = seg_frame.shape[0]
            seg_width = seg_frame.shape[1]
            for jdx in range(seg_width):
                for idx in range(seg_height): 
                    curr_rgb = [seg_frame[idx, jdx, :]]  
                    curr_rgb_tuple = [tuple(e) for e in curr_rgb]
                    if curr_rgb_tuple[0] == (0, 0, 255):
                        color_to_id[curr_rgb_tuple[0]] = 'DeadPixel'
                    curr_id = color_to_id[curr_rgb_tuple[0]]
                    for obj in curr_objects:
                        if curr_id in obj['objectId']:
                            if obj['pickupable']:
                                affordance_mask[0, idx, jdx] = 255
                                # affordance_mask[1, idx, jdx] = 0
                            elif (obj['receptacle'] and not obj['openable']) or (obj['receptacle'] and (obj['openable'] and obj['isOpen'])):
                                # affordance_mask[0, idx, jdx] = 0
                                affordance_mask[1, idx, jdx] = 255
            
            img = np.array(state.frame, dtype=np.uint8)                                                         
            img_channel_first = np.moveaxis(img, -1, 0)
            obs = np.concatenate((img_channel_first, affordance_mask), axis=0)
        elif self.observations == "rgbda":
            img = np.array(state.frame, dtype=np.uint8)
            img_channel_first = np.moveaxis(img, -1, 0)

            curr_objects = state.metadata['objects']
            num_obj_actions = 2 # [take, put]
            affordance_mask = np.zeros((num_obj_actions, self.args.obs_size, self.args.obs_size), dtype=np.uint8)
            seg_frame = state.instance_segmentation_frame

            color_to_id = state.color_to_object_id
            seg_height = seg_frame.shape[0]
            seg_width = seg_frame.shape[1]
            for jdx in range(seg_width):
                for idx in range(seg_height): 
                    curr_rgb = [seg_frame[idx, jdx, :]]  
                    curr_rgb_tuple = [tuple(e) for e in curr_rgb]
                    if curr_rgb_tuple[0] == (0, 0, 255):
                        color_to_id[curr_rgb_tuple[0]] = 'DeadPixel'
                    curr_id = color_to_id[curr_rgb_tuple[0]]
                    for obj in curr_objects:
                        if curr_id in obj['objectId']:
                            if obj['pickupable']:
                                affordance_mask[0, idx, jdx] = 255
                                # affordance_mask[1, idx, jdx] = 0
                            elif (obj['receptacle'] and not obj['openable']) or (obj['receptacle'] and (obj['openable'] and obj['isOpen'])):
                                # affordance_mask[0, idx, jdx] = 0
                                affordance_mask[1, idx, jdx] = 255
            
            img = np.array(state.frame, dtype=np.uint8)                                                         
            img_channel_first = np.moveaxis(img, -1, 0)
            depth_frame = state.depth_frame[np.newaxis, ...]*50.0
            obs = np.concatenate((img_channel_first, depth_frame, affordance_mask), axis=0)
        else:
            raise NotImplementedError

        return obs

    def agent_pose(self, state):
        agent = state.metadata['agent']
        pose = (agent['position']['x'], agent['position']['y'], agent['position']['z'],
                self.rot_size_y*np.round(agent['rotation']['y']/self.rot_size_y),
                self.rot_size_x*np.round(agent['cameraHorizon']/self.rot_size_x))
        return pose

    # return last event every time state is called
    @property
    def state(self):
        return self.controller.last_event

    @property
    def inv_obj(self):
        inventory = self.state.metadata['inventoryObjects']
        return inventory[0]['objectId'] if len(inventory) > 0 else None

    # action functions
    def move(self, direction):
        act_params = dict(action='MoveAhead')
        return {'params': act_params}

    def look(self, direction):
        rotation = self.state.metadata['agent']['rotation']['y']
        horizon = self.state.metadata['agent']['cameraHorizon']
        if direction=='up':
            horizon -= self.rot_size_x
        elif direction=='down':
            horizon += self.rot_size_x

        act_params = dict(action='RotateLook', rotation=rotation, horizon=horizon)

        return {'params': act_params}

    def turn(self, direction):
        rotation = self.state.metadata['agent']['rotation']['y']
        horizon = self.state.metadata['agent']['cameraHorizon']
        if direction=='right':
            rotation += self.rot_size_y
        elif direction=='left':
            rotation -= self.rot_size_y

        act_params = dict(action='RotateLook', rotation=rotation, horizon=horizon)

        return {'params': act_params}

    def take(self, direction):
        obj_property = lambda obj: obj['pickupable']
        target_obj = self.get_target_obj(obj_property)

        act_params = None
        if target_obj['objectId'] is not None:
            act_params = dict(action='PickupObject', objectId=target_obj['objectId'], 
                            forceAction=False, manualInteract=False)

        return {'params': act_params}


    def put(self, direction):
        obj_property = lambda obj: obj['receptacle'] and (obj['openable'] and obj['isOpen'] or not obj['openable'])
        target_obj = self.get_target_obj(obj_property, overlap_thresh=0.1) # easier to put things down

        act_params = None
        if target_obj['objectId'] is not None:
            act_params = dict(action='PutObject', objectId=target_obj['objectId'], forceAction=True, placeStationary=True)

        return {'params': act_params}

    def act(self, action):
        action_info = {'action': action, 'success':False}
        action_info.update(self.action_functions[action](action))

        if action_info['params'] is not None:
            self.controller.step(action_info['params'])
            action_info['success'] = self.state.metadata['lastActionSuccess']
        
        # if it's a movement action, double check that you're still on the grid
        if action_info['action']=='forward' and action_info['success']:
            x, y, z, rot, hor = self.agent_pose(self.state)
            if (x, y, z) not in self.reachable_positions:
                gpos = min(self.reachable_positions, key=lambda p: (p[0]-x)**2 + (p[2]-z)**2)
                self.controller.step(dict(action='TeleportFull', position=dict(x=gpos[0], y=gpos[1], z=gpos[2]), 
                                    rotation=dict(x=0, y=rot, z=0), horizon=hor, standing=True))            

        return action_info
    
    # initialize the environment for kb agent
    def init_env_debug(self):
        self.scene = self.debug_scene
        self.episode = self.debug_episode

        self.controller.reset(self.scene)
        self.controller.step(dict(action='Initialize', **self.init_params))
        self.controller.step(dict(action='GetReachablePositions'))
        self.reachable_positions = set([(pos['x'], pos['y'], pos['z']) for pos in self.state.metadata['actionReturn']])
        self.controller.step(dict(action='TeleportFull', position=dict(x=0, y=0.9009991, z=2.25), rotation=dict(x=0, y=180, z=0), horizon=0, standing=True))

    # initialize the environment
    def init_env(self):
        self.t=0
        if self.mode == "train":
            self.scene = self.rs.choice(['FloorPlan%d'%idx for idx in range(6, 30+1)]) # 6 --> 30 = train. 1 --> 5 = test
            self.episode = self.rs.randint(1000000000)
        elif self.mode == "eval":
            self.scene, self.episode = next(self.eval_scene_iter,None), next(self.eval_episode_iter,None)
            if self.scene is None:
                self.eval_episode_iter = iter(self.eval_episodes)
                self.eval_scene_iter = iter(self.eval_scenes)
                self.scene, self.episode = next(self.eval_scene_iter), next(self.eval_episode_iter)
        elif self.mode =="debug":
            self.init_env_debug()
            return
        else:
            raise NotImplementedError

        self.controller.reset(scene=self.scene)
        self.controller.step(dict(action='Initialize', **self.init_params))
       
        # randomize object locations
        self.controller.step(dict(action='InitialRandomSpawn',
                randomSeed=self.episode,
                forceVisible=True,
                numPlacementAttempts=5))
 
        # borrowed from Facebook's interaction-exploration
        self.controller.step(dict(action='GetReachablePositions'))
        reachable_positions = [(pos['x'], pos['y'], pos['z']) for pos in self.state.metadata['actionReturn']]
        
        init_rs = np.random.RandomState(self.episode)
        rot = init_rs.choice([i*self.rot_size_y for i in range(360//self.rot_size_y)])
        pos = reachable_positions[init_rs.randint(len(reachable_positions))]
        self.controller.step(dict(action='TeleportFull', position=dict(x=pos[0], y=pos[1], z=pos[2]), 
                            rotation=dict(x=0, y=rot, z=0), horizon=0, standing=True))
        
        self.reachable_positions = set(reachable_positions)

        # reward specific parameters
        if self.reward_type == "interaction_count":
            self.interaction_count = collections.defaultdict(int)
        if self.reward_type == "interaction_navigation":    
            self.interaction_count = collections.defaultdict(int)
            self.camera_poses = collections.defaultdict(int)
