from envs.thor_evaluate import ThorEvaluateEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from envs.config.config import config_parser
from envs.thor_map_evaluate import ThorMapEvaluateEnv
from bridge import Bridge
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

set_random_seed(seed=7) #51184165 or 6934152

# ROS imports
import rospy

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

saved_model = PPO.load(saved_model_path, env=env)
evaluate_policy(saved_model, saved_model.get_env(), n_eval_episodes=1, deterministic=False)