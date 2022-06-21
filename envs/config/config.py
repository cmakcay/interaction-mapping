import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()

    # thor env options
    parser.add_argument("--observations", type=str, default="rgba", help="rgb, rgbd, rgba, rgbda")
    parser.add_argument("--action_type", type=str, default="with_int", help="with_int, without_int")
    parser.add_argument("--x_display", type=float, default=0.0)
    parser.add_argument("--headless", type=bool, default=False)
    parser.add_argument("--local_exe", type=str, default="None", help="path to local executable file")
    # parser.add_argument("--local_exe", type=str, default="/home/asl/plr/ai2thor/unity/builds/test3.x86_64", help="path to local executable file")
    parser.add_argument("--obs_size", type=int, default=80, help="observation frame size")
    parser.add_argument("--rot_size_x", type=int, default=15)
    parser.add_argument("--rot_size_y", type=int, default=30)
    parser.add_argument("--frame_size", type=int, default=80, help="size of image frames")
    parser.add_argument("--num_steps", type=int, default=400, help="when each episode ends")
    parser.add_argument("--eval_scenes", nargs="+", default=["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan1", "FloorPlan2", "FloorPlan3"])
    parser.add_argument("--eval_episodes", nargs="+", default=[12345, 31452, 52314, 41235, 25431, 72583, 67213, 93275])
    parser.add_argument("--reward_type", type=str, default="interaction_navigation")
    parser.add_argument('--debug_scene', type=str, default='FloorPlan1')
    parser.add_argument('--debug_episode', type=int, default=0)
    parser.add_argument('--eval_mode', type=str, default='mapper', help='thor or mapper')

    # training options
    parser.add_argument("--num_processes", type=int, default=4)

    # kb agent options
    parser.add_argument('--csv_path', default='/home/asl/plr/kb_agent_dataset/groundtruth_labels_debug.csv')
    parser.add_argument('--save_path', default='/home/asl/plr/kb_agent_dataset/run_debug')

    return parser
