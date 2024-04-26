from typing import Set, cast

import torch
import torch.nn as nn
import torchfilter
import torchfilter.types as types
from fannypack.nn import resblocks

from ..tasks import KittiTask 
from . import layers
from .dynamics import KittiDynamicsModel 

class KittiParticleFilter(torchfilter.filters.ParticleFilter, KittiTask.Filter):
    def __init__(self):
        """Initializes a particle filter for our door task."""

        super().__init__(
            dynamics_model=KittiDynamicsModel(),
            measurement_model=KittiMeasurementModel(),
            num_particles=30,
        )

    def train(self, mode: bool = True):
        """Adjust particle count based on train vs eval mode."""
        self.num_particles = 30 if mode else 300
        super().train(mode)

        
class KittiMeasurementModel(torchfilter.base.ParticleFilterMeasurementModel): 
    def __init__(
        self, units: int = 64, modalities: Set[str] = {"image", "gps"} 
    ): 
        """Initializes a measurement model for our kitti task"""
        super().__init__(state_dim = 5)

        valid_modalities = {"image", "gps"} 
        assert len(valid_modalities | modalities) == 2, "Received invalid modality"
        assert len(modalities) > 0, "Received empty modality list"
        
        self.modalities = modalities
        
        if "image" in modalities: 
            self.observation_image_layers = layers.observation_image_layers(units)
        if "gps" in modalities: 
            self.observation_gps_layers = layers.observation_gps_layers(units)

        self.state_layers = layers.state_layers(units)
        self.shared_layers = nn.Sequential(
            nn.Linear(units * (1 + len(modalities)), units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
            resblocks.Linear(units),
            nn.Linear(units, 1),
            # nn.LogSigmoid()
        )

        self.units = units
        
    def forward(
        self, *, states: types.StatesTorch, observations: types.ObservationsTorch
    ) -> types.StatesTorch:
        assert type(observations) == dict
        assert len(states.shape) == 3  # (N, M, state_dim)
        assert states.shape[2] == self.state_dim
        observations = cast(types.TorchDict, observations)

        # N := distinct trajectory count
        # M := particle count
        N, M, _ = states.shape

        # Construct observations feature vector
        # (N, obs_dim)
        
        obs = []
        if "image" in self.modalities:
            stacked_raw_image_and_diff = torch.cat(
                (observations["raw_image"], observations["difference_image"]), dim=1
            )
            obs.append(
                self.observation_image_layers(stacked_raw_image_and_diff)
            )
        if "gps" in self.modalities:
            total_gps = torch.cat((observations["gps_fv"], observations["gps_av"]), dim=1)
            obs.append(self.observation_gps_layers(total_gps))
            
        observation_features = torch.cat(obs, dim=1)

        # (N, obs_features) => (N, M, obs_features)
        observation_features = observation_features[:, None, :].expand(
            N, M, self.units * len(obs)
        )
        assert observation_features.shape == (N, M, self.units * len(obs))

        # (N, M, state_dim) => (N, M, units)
        state_features = self.state_layers(states)
        # state_features = self.state_layers(states * torch.tensor([[[1., 0.]]], device=states.device))
        assert state_features.shape == (N, M, self.units)

        # BUG: Velocities from GPS and state are being combined. 
        merged_features = torch.cat((observation_features, state_features), dim=2)
        assert merged_features.shape == (N, M, self.units * (len(obs) + 1))

        # (N, M, merged_dim) => (N, M, 1)
        log_likelihoods = self.shared_layers(merged_features)
        assert log_likelihoods.shape == (N, M, 1)

        # Return (N, M)
        return torch.squeeze(log_likelihoods, dim=2)
