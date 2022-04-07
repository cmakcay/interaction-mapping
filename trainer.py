from envs.thor import ThorEnv
from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ThorEnv(mode="train", seed=rank)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


if __name__=='__main__':

    num_processes = 8

    # evaluation environment
    eval_env = ThorEnv("eval", 2336435)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=5000,
                             deterministic=True, render=False)

    # train environment
    train_env = SubprocVecEnv([make_env(i) for i in range(num_processes)])

    model = PPO2(CnnLnLstmPolicy, train_env, nminibatches=2, verbose=1, tensorboard_log="./logs/tensorboard_log/")
    model.learn(total_timesteps=500000, callback = eval_callback)
