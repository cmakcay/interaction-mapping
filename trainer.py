from envs.thor import ThorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from models.CustomCNN import CustomCNN
from stable_baselines3.common.monitor import Monitor
from bridge import Bridge
from envs.config.config import config_parser

# ROS imports
import rospy

def make_env(rank, nh, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ThorEnv(mode="debug", seed=seed + 173127*rank, nh=nh)
        env.seed(seed + 173127*rank)
        return Monitor(env)

    return _init

def make_eval_env(seed, nh):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ThorEnv(mode="eval", seed=seed, nh=nh)
        env.seed(seed)
        return Monitor(env)

    return _init


if __name__=='__main__':

    parser = config_parser()
    args = parser.parse_args()
    rospy.init_node("mapper")

    num_processes = args.num_envs
    mappers = []
    for i in range(num_processes):        
        mappers.append(Bridge(env_index = i))



    # evaluation environment
    # eval_env = SubprocVecEnv([make_eval_env(seed=2336435, nh=mapper)])
    # eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
    #                          log_path='./logs/', eval_freq=2560,
    #                          deterministic=False, render=False, n_eval_episodes=8)

    # train environment
    # train_env = ThorEnv(mode='debug', seed=73745, nh=mapper)
    # train_env = SubprocVecEnv([make_env(rank=i, nh=mapper) for i in range(num_processes)])    
    train_env = DummyVecEnv([make_env(rank=i, nh=mappers[i]) for i in range(num_processes)])    
    
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[512, dict(vf=[512], pi=[512])],
    )
    
    model = PPO("CnnPolicy", train_env, batch_size=128, verbose=1, n_steps=256, n_epochs=4, 
                learning_rate=1e-4, tensorboard_log="./logs/tensorboard_log/", ent_coef=0.01, policy_kwargs=policy_kwargs)
    
    model.learn(total_timesteps=1000000)
    # model.learn(total_timesteps=1000000, callback = eval_callback)
    model.save("FinalModel")
