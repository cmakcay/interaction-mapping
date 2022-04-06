from envs.thor import ThorEnv
from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.vec_env import SubprocVecEnv

if __name__=='__main__':

    # evaluation environment
    eval_env = ThorEnv("eval")
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1000,
                             deterministic=True, render=False)

    # train environment
    train_env = make_vec_env(ThorEnv, n_envs=2, seed=0, env_kwargs = {"mode": "train"}, vec_env_cls=SubprocVecEnv)

    model = PPO2(CnnLnLstmPolicy, train_env, nminibatches=2, verbose=1, tensorboard_log="./logs/tensorboard_log/")
    model.learn(total_timesteps=250000, callback = eval_callback)
