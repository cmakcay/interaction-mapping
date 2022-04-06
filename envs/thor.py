import collections
import gym
from gym import spaces
import ai2thor.controller
from .config.config import config_parser
import numpy as np
from PIL import Image
import itertools

class ThorEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, mode):
        super(ThorEnv, self).__init__()

        # get args
        parser = config_parser()
        self.args = parser.parse_args()

        # set gym env args
        x_display = self.args.x_display
        local_exe = None if self.args.local_exe=='None' else self.config.local_exe
        obs_size = self.args.obs_size #switch to a smaller size than 300x300?
        num_channels = self.args.num_channels #3 for rgb but will probably change with affordance

        # set global args for the class
        self.init_params = {
            'gridSize': 0.25,
            'renderDepthImage': True,
            'renderInstanceSegmentation': True,
            'visibilityDistance': 1.0, #maybe switch to 1.5?
        }
        self.mode = mode #train or eval
        self.rot_size_x = self.args.rot_size_x
        self.rot_size_y = self.args.rot_size_y
        self.frame_sz = self.args.frame_size
        self.max_t = self.args.num_steps
        self.rs = np.random.RandomState(self.args.random_seed) #set random seed for same training data
        self.eval_episodes = iter(self.args.eval_episodes) # use predefined scenes and episodes for evaluation
        self.eval_scenes = iter(self.args.eval_scenes)
        self.reward_type = self.args.reward_type

        # only navigation + take and put
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
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(obs_size, obs_size, num_channels), dtype=np.uint8)

        self.N = 5
        self.center = ((self.frame_sz//self.N)*(self.N//2), (self.frame_sz//self.N)*(self.N+1)//2)
        self.center_grid = np.array([[self.center[0], self.center[0], self.center[1], self.center[1]]]) # xyxy     

        # create ai2thor controller
        self.controller = ai2thor.controller.Controller(quality='Ultra',
                                                        local_executable_path=local_exe,
                                                        x_display=x_display)




    # functions need to be overwritten for gym env
    def step(self, action):
        self.t += 1
        self.step_info = self.act(self.actions[action])
        observation = self.get_observation(self.state)
        reward = self.get_reward()
        done = self.get_done()
        return observation, reward, done, self.step_info

    def reset(self):
        self.t = 0
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
        else:
            raise NotImplementedError

    def get_done(self):
        return self.t>=self.max_t
    
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


    def get_observation(self, state):
        img = state.frame
        return img

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

    # initialize the environment
    def init_env(self):
        if self.mode == "train":
            self.scene = self.rs.choice(['FloorPlan%d'%idx for idx in range(6, 30+1)]) # 6 --> 30 = train. 1 --> 5 = test
            self.episode = self.rs.randint(1000000000)
        elif self.mode == "eval":
            self.scene, self.episode = next(self.eval_scenes), next(self.eval_episodes)
            print ('INIT: %s episode %s'%(self.scene, self.episode))
        else:
            raise NotImplementedError

        self.controller.reset(scene=self.scene, **self.init_params)
       
        # randomize objects (should we do it?)
        self.controller.step(dict(action='InitialRandomSpawn',
                randomSeed=self.episode,
                forceVisible=True,
                numPlacementAttempts=5))

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