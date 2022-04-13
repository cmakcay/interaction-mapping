import gym
import torch
from torchvision import transforms
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # self.transform_image = transforms.Compose([transforms.Resize([80, 80]), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform_image = transforms.Compose([transforms.Resize([80, 80])])


        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.Flatten(),
        )

        n_flatten = 1152
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU(True))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.transform_image(observations)
        observations = self.cnn(observations)
        observations = self.linear(observations)
        return observations