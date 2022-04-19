source /home/asl/catkin_ws/devel/setup.bash
export num_envs=2

for (( i=0; i<$num_envs; i++ ))
do
gnome-terminal -- roslaunch panoptic_mapping_ros thor_simultaneous.launch env_index:=$i
sleep 1
done

source /home/asl/plr/interaction-mapping/venv/bin/activate
python trainer.py --num_envs $num_envs
