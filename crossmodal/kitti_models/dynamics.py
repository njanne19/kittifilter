import torch
import torch.nn as nn
import torchfilter
import torchfilter.types as types
from fannypack.nn import resblocks

from . import layers


class KittiDynamicsModel(torchfilter.base.DynamicsModel):
    def __init__(self, units=64):
        """Initializes a dynamics model for our door task."""

        # Here, state dim is 2. One for forward velocity, one for angular velocity 
        super().__init__(state_dim=2)

        # Control dim is 13, three are forward/lateral/upward acceleration, nine are are a flattened rotation matrix, one is timestep   
        control_dim = 13

        # Fixed dynamics covariance
        self.Q_scale_tril = nn.Parameter(
            torch.linalg.cholesky(torch.diag(torch.FloatTensor([0.05, 0.05]))),
            requires_grad=False,
        )

        # Build the neural network 
        self.state_layers = layers.state_layers(units=units) 
        self.control_layers = layers.control_layers(units=units)
        self.shared_layers = nn.Sequential(
            nn.Linear(units * 2, units),
            resblocks.Linear(units),
            resblocks.Linear(units),
            resblocks.Linear(units),
            nn.Linear(units, self.state_dim),
        )
        self.units = units

    def forward(
        self,
        *,
        initial_states: types.StatesTorch,
        controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        N, state_dim = initial_states.shape[:2]
        assert state_dim == self.state_dim

        # (N, control_dim) => (N, units // 2)
        control_features = self.control_layers(controls)
        
        # (N, state_dim) => (N, units // 2)
        state_features = self.state_layers(initial_states)
        
        # (N, units)
        merged_features = torch.cat((control_features, state_features), dim=-1)
        
        # (N, units * 2) => (N, state_dim)
        output_features = self.shared_layers(merged_features)
        
        # We separately compute a direction for our network and a scalar "gate"
        # These are multiplied to produce our final state output
        state_update_direction = output_features
        state_update_gate = torch.sigmoid(output_features[..., -1:])
        state_update = state_update_direction * state_update_gate

        # Return residual-style state update, constant uncertainties
        states_new = initial_states + state_update
        scale_trils = self.Q_scale_tril[None, :, :].expand(N, state_dim, state_dim)
        return states_new, scale_trils
     
