import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()

    # thor env options
    parser.add_argument("--x_display", type=float, default=0.0)
    parser.add_argument("--local_exe", type=str, default="None", help="path to local executable file")
    parser.add_argument("--obs_size", type=int, default=300, help="observation frame size")
    parser.add_argument("--num_channels", type=int, default=3, help="number of channels")
    parser.add_argument("--rot_size_x", type=int, default=15)
    parser.add_argument("--rot_size_y", type=int, default=30)
    parser.add_argument("--frame_size", type=int, default=300, help="size of image frames")
    parser.add_argument("--num_steps", type=int, default=256, help="when each episode ends")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed for reproducibility")
    parser.add_argument("--eval_scenes", nargs="+", default=["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"])
    parser.add_argument("--eval_episodes", nargs="+", default=[12345, 31452, 52314, 41235, 25431])
    parser.add_argument("--reward_type", type=str, default="interaction_count")

    # training options
    parser.add_argument("--num_processes", type=int, default=4)

    return parser
