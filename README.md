# Interactive Exploration for Mapping

This repo contains code to train an agent to navigate in an unknown environment and interact with objects to perform more complete object-level mapping and to evaluate the completeness of the resulting object-level maps which are created using the panoptic mapping framework introduced in [Panoptic Multi-TSDFs](https://arxiv.org/abs/2109.10165). The reinforcement learning framework is inspired by [Learning Affordance Landscapes for Interaction Exploration in 3D Environments](https://arxiv.org/pdf/2008.09241.pdf).

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
We use ThorEnv in envs/thor.py in training. The parameters that are used in training can be found in envs/config/config.py. For training:

1. Select the observation space using the *observations* parameter. There are four options: *rgb* (color images), *rgbd* (color and depth images), *rgba* (color images + ground truth affordances), and *rgbda* (color and depth images + ground truth affordances). The default value is *rgba*.

2. Select the action space using the *action_type* parameter. There are two options: *with_int* (the agent can interact with objects) and *without_int* (the agent cannot interact with objects).

3. Set the *headless* parameter to *True* if you want to use headless rendering.

4. Select the reward type using the *reward_type* parameter. There are two options: *interaction_count* and *interaction_navigation*. The latter is used to train our agents. The weights for interaction and navigation reward can be set in get_reward() function of envs/thor.py whose default values are 1.0 and 0.2, respectively.

5. Set the number of parallel environments to run in trainer.py whose default value is 12.

After setting up the parameters, start training: 
```
python trainer.py
```

The tensorboard logs are saved under a folder called *logs* in the same directory. The best model (best_model.zip) can be found here when training ends.

## Bridge to Mapping Framework
1. Follow the instructions in the mapper [repo](https://github.com/ikaftan/panoptic_mapping) to build and source it.

2. Download the CSV files containing the ground truth labels from [here](https://drive.google.com/drive/folders/1Uf3HPTYpzWdVD-dzeUutLp2a-l8MU7ob?usp=sharing) which are used by the mapper.
 
3. Test the bridge with a keyboard agent that simultaneously maps the environment by going through the following steps:
- Set the *csv_path* parameter in envs/config/config.py to the path of *groundtruth_labels_debug.csv* file.
- Change the path in line 19 of *thor.yaml* file located in src/panoptic_mapping/panoptic_mapping_ros/config/mapper to the path of *groundtruth_labels_debug.csv* file.
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
There are three evaluation metrics: object coverage, position coverage, and the number of observed voxels of individual objects. The parameters that are used in evaluation can be found in envs/config/config.py. The trained models can be downloaded from [here](https://drive.google.com/drive/folders/1orKThDW-8UfKkpVmDTYiPYkmTBYsoURv?usp=sharing).

1. Select the evaluation mode using the *eval_mode* parameter. There are two options: *thor* and *mapper*. The first two metrics are obtained by the simulator and the last metric is obtained from the mapper.

2. We use ThorEvaluateEnv in envs/thor_evaluate.py for *thor* option:
- There are 50 randomly selected evaluation episodes listed in envs/config/evaluation_list.csv which are used in obtaning object coverage and position coverage.
- Select the action space using the *action_type* parameter depending on the the trained model.
- Set the number of evaluation steps using the *num_steps* parameter. The default value is 1024 for these metrics.
- Change line 31 of *evaluate_model.py* file to the path of *best_model.zip* of the desired model.
- Create a folder called *fraction_logs* under the same directory. The metrics will be saved here with names interaction_log.csv and position_log.csv.
- Set n_eval_episodes to 50 in line 30 of *evaluate_model.py* for these metrics.
- Evaluate the trained model:
```
python evaluate_model.py
```
- The values can then be plotted using a script similar to fraction_logs.py after specifying the paths of interaction_log.csv and position_log.csv in lines 5-6:
```
python fraction_logs.py
```

3. We use ThorMapEvaluateEnv in envs/thor_map_evaluate.py for *mapper* option:
- There are 50 randomly selected evaluation episodes listed in envs/config/evaluation_list.csv. We use one episode at a time to obtain the number of observed voxels of an object present in that scene. The ground truth labels of these episodes are located under *eval_labels*.
- Specify the selected scene in line 43 and the selected episode in line 44 of envs/thor_map_evaluate.py and change line 78 to the path of groundtruth_labels_{eval_scene}_{eval_episode}.csv in the same script.
- Select the action space using the *action_type* parameter depending on the the trained model.
- Set the number of evaluation steps using the *num_steps* parameter. The default value is 400 for this metric.
- Change line 31 of *evaluate_model.py* file to the path of *best_model.zip* of the desired model.
- Set n_eval_episodes to 1 in line 30 of *evaluate_model.py* for this metric since we evaluate on one episode at a time.
- Change the path in line 19 of *thor_0.yaml* file in src/panoptic_mapping/panoptic_mapping_ros/config/eval to the path of *groundtruth_labels_{eval_scene}_{eval_episode}.csv* file, the id in line 12 to the id of the selected object which can be found in the csv file, and the episode length in line 15 to the number of evaluation steps mentioned before.
- Launch the mapper:
```
roslaunch panoptic_mapping_ros thor_eval.launch
```
- Evaluate the trained model:
```
python evaluate_model.py
```
- You can use rqt_multiplot to plot the number of observed voxels of the selected object simultaneously:
```
rosrun rqt_multiplot rqt_multiplot
```
