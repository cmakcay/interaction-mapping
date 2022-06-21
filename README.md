# Interactive Exploration for Mapping

This repo contains code to train an agent to navigate in an unknown environment and interact with objects to perform more complete object-level mapping and to evaluate the completeness of the resulting object-level maps which are created using the panoptic mapping framework introduced in [Panoptic Multi-TSDFs](https://github.com/ethz-asl/panoptic_mapping). 

## Installation

1. Download repo using SSH or HTTPS:
```
git clone git@github.com:cmakcay/interaction-mapping.git  # SSH
git clone https://github.com/cmakcay/interaction-mapping.git # HTTPS
```

3. Create a virtual environment and source it:
```
python -m venv venv
source venv/bin/activate
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download the simulator files (one-time download when [AI2-iTHOR](https://ai2thor.allenai.org/ithor/documentation) is first run) and test the simulator with a simple keyboard agent:
```
python kb_agent.py
```

## Training
The parameters that are used in training and evaluation can be found in envs/config/config.py. For training:

1. Select the observation space using the *observations* parameter. There are four options: *rgb* (color images), *rgbd* (color and depth images), *rgba* (color images + ground truth affordances), and *rgbda* (color and depth images + ground truth affordances). The default value is *rgba*.

2. Select the action space using the *action_type* parameter. There are two options: *with_int* (the agent can interact with objects) and *without_int* (the agent cannot interact with objects).

3. Set the *headless* parameter to *True* if you want to use headless rendering.

4. Select the reward type using the *reward_type* parameter. There are two options: *interaction_count* and *interaction_navigation*. The latter is used to train our agents. The weights for interaction and navigation reward can be set in get_reward() function of envs/thor.py whose default values are 1.0 and 0.2, respectively.

5. Set the number of parallel environments to run in trainer.py whose default value is 12.

After setting up the parameters, start training: 
```
python trainer.py
```

## Bridge to Mapping Framework
1. Follow the instructions in the mapper [repo](https://github.com/ikaftan/panoptic_mapping) to build and source it.

2. Download the CSV files containing the ground truth labels from [here](https://drive.google.com/drive/folders/1Uf3HPTYpzWdVD-dzeUutLp2a-l8MU7ob?usp=sharing) which are used by the mapper.
 
3. Test the bridge with a keyboard agent that simultaneously maps the environment by going through the following steps:
- Set the *csv_path* parameter in envs/config/config.py to the path of *groundtruth_labels_debug.csv* file.
- Change line 19 of *thor.yaml* file located in src/panoptic_mapping/panoptic_mapping_ros/config/mapper to the path of *groundtruth_labels_debug.csv* file.
- Open the mapper terminal and launch the mapper:
```
roslaunch panoptic_mapping_ros thor_kb_agent.launch
```
- Open the simulator terminal and run the keyboard agent:
```
python kb_simultaneous.py
```

You will see that the map is incrementally built in RViz as you navigate in the environment using the keyboard agent.

## Evaluation
