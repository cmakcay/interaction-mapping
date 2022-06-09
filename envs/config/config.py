import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()

    # thor env options
    parser.add_argument("--observations", type=str, default="rgba", help="rgb, rgbd, rgba")
    parser.add_argument("--x_display", type=float, default=0.0)
    parser.add_argument("--headless", type=bool, default=False)
    parser.add_argument("--local_exe", type=str, default="None", help="path to local executable file")
    # parser.add_argument("--local_exe", type=str, default="/home/asl/plr/ai2thor/unity/builds/test3.x86_64")
    parser.add_argument("--obs_size", type=int, default=80, help="observation frame size")
    parser.add_argument("--rot_size_x", type=int, default=15)
    parser.add_argument("--rot_size_y", type=int, default=30)
    parser.add_argument("--frame_size", type=int, default=400, help="size of image frames")
    parser.add_argument("--num_steps", type=int, default=256, help="when each episode ends")
    parser.add_argument("--eval_scenes", nargs="+", default=["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan1", "FloorPlan2", "FloorPlan3"])
    parser.add_argument("--eval_episodes", nargs="+", default=[12345, 31452, 52314, 41235, 25431, 72583, 67213, 93275])
    parser.add_argument("--reward_type", type=str, default="interaction_count")
    parser.add_argument('--debug_scene', type=str, default='FloorPlan4')
    parser.add_argument('--debug_episode', type=int, default=128984)

    # training options
    parser.add_argument("--num_train_envs", type=int, default=3)
    parser.add_argument("--num_eval_envs", type=int, default=5)

    # kb agent options
    parser.add_argument('--csv_path', default='/home/asl/plr/kb_agent_dataset/groundtruth_labels_debug.csv')
    parser.add_argument('--save_path', default='/home/iremkaftan/Desktop/kb_agent_dataset/run1')

    return parser
