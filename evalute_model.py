from envs.thor_evaluate import ThorEvaluateEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from envs.config.config import config_parser
from envs.thor_map_evaluate import ThorMapEvaluateEnv
from bridge import Bridge
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

set_random_seed(seed=7)

# ROS imports
import rospy

# def make_eval_env(seed):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: (str) the environment ID
#     :param num_env: (int) the number of environments you wish to have in subprocesses
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     """
#     def _init():
#         env = ThorEvaluateEnv(seed=seed)
#         env.seed(seed)
#         return Monitor(env)
#     return _init

saved_model_path = "/home/asl/plr/backups/int_nav_0.2/best_model.zip"
parser = config_parser()
args = parser.parse_args()

if args.eval_mode == 'thor':
    env = ThorEvaluateEnv(seed=0)
elif args.eval_mode == 'mapper':
    rospy.init_node("mapper")

    # create node handle
    mapper = Bridge(env_index=0)
    env = ThorMapEvaluateEnv(seed=0, nh=mapper)

# # Might not need this dict in all cases
# custom_objects = {
#     "lr_schedule": lambda x: .003,
#     "clip_range": lambda x: .02
# }
# saved_model = PPO.load(saved_model_path, custom_objects=custom_objects, env=env)

saved_model = PPO.load(saved_model_path, env=env)
evaluate_policy(saved_model, saved_model.get_env(), n_eval_episodes=1, deterministic=False)