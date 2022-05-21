from envs.thor_evaluate import ThorEvaluateEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

saved_model_path = "/home/asl/plr/backups/int_nav_0.2/best_model.zip"
env = ThorEvaluateEnv(seed=0)
saved_model = PPO.load(saved_model_path, env=env)
evaluate_policy(saved_model, saved_model.get_env(), n_eval_episodes=50, deterministic=False)