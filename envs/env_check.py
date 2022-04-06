from thor import ThorEnv
from stable_baselines.common.env_checker import check_env

env = ThorEnv("train")
check_env(env)