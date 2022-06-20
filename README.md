# Interactive Exploration for Mapping

This repo contains code to train an agent to navigate in an unknown environment and interact with objects to perform more complete object-level mapping and to evaluate the completeness of the resulting object-level maps which are created using the panoptic mapping framework introduced in [Panoptic Multi-TSDFs](https://github.com/ethz-asl/panoptic_mapping). 

## Installation

1. Download repo using SSH or HTTPS:
```
git clone git@github.com:cmakcay/interaction-mapping.git  # SSH
git clone https://github.com/cmakcay/interaction-mapping.git # HTTPS
```

2. Install required packages:
Install required packages:
```
pip install -r requirements.txt
```

3. Download the simulator files (one-time download when [AI2-iTHOR](https://ai2thor.allenai.org/ithor/documentation) is first run) and test the simulator with a simple keyboard agent:
```
python kb_agent.py
```

## Training
The parameters that are used in training and evaluation can be found in envs/config/config.py. For training:

1. Select the observation space using the *observations* parameters. There are four options: *rgb* (color images), *rgbd* (color and depth images), *rgba* (color images + ground truth affordances), and *rgbda* (color and depth images + ground truth affordances). The default value is *rgba*.

2. Set the *headless* parameter to *True* if you want to use headless rendering.

3. Select the reward type using the *reward_type* parameter. There are two options: *interaction_count* and *interaction_navigation*. The latter is used to train our agents. The weights for interaction and navigation reward can be set in get_reward() function of envs/thor.py whose default values are 1.0 and 0.2, respectively.

4. The action space can be selected in the init() function of envs/thor.py. There are two options: *with interaction* (comment line 84) and *without interaction* (comment line 83).

5. The number of parallel environments to run can be set in trainer.py whose default value is 12.

After setting up the parameters, start training: 
```
python trainer.py
```

## Bridge to Mapping Framework
1. Follow the instructions in the mapper [repo](https://github.com/ikaftan/panoptic_mapping) to set it up. 

2. Download the CSV files containing the ground truth labels from [here](https://drive.google.com/drive/folders/1Uf3HPTYpzWdVD-dzeUutLp2a-l8MU7ob?usp=sharing).
