source /home/asl/catkin_ws/devel/setup.bash
export num_envs=8

for (( i=0; i<$num_envs; i++ ))
do
gnome-terminal -- roslaunch panoptic_mapping_ros thor_train.launch env_index:=$i
sleep 1
done

source /home/asl/plr/interaction-mapping/venv/bin/activate
python trainer.py
