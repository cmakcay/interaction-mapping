from envs.thor import ThorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from models.CustomCNN import CustomCNN
from stable_baselines3.common.monitor import Monitor


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ThorEnv(mode="train", seed=seed + 173127*rank)
        env.seed(seed + 173127*rank)
        return Monitor(env)

    return _init

def make_eval_env(seed):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ThorEnv(mode="eval", seed = seed)
        env.seed(seed)
        return Monitor(env)

    return _init


if __name__=='__main__':

    num_processes = 12

    # evaluation environment
    eval_env = SubprocVecEnv([make_eval_env(seed=2336435)])
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=2560,
                             deterministic=False, render=False, n_eval_episodes=8)

    # train environment
    train_env = SubprocVecEnv([make_env(i) for i in range(num_processes)])    
        
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[512, dict(vf=[512], pi=[512])],
    )
    
    model = PPO("CnnPolicy", train_env, batch_size=128, verbose=1, n_steps=256, n_epochs=4, 
                learning_rate=1e-4, tensorboard_log="./logs/tensorboard_log/", ent_coef=0.01, policy_kwargs=policy_kwargs)
    

    model.learn(total_timesteps=700000, callback = eval_callback)
    model.save("FinalModel")
