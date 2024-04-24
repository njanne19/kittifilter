import torch
import torch.nn as nn
import torchfilter
import torchfilter.types as types
from fannypack.nn import resblocks

from . import layers


class KittiDynamicsModel(torchfilter.base.DynamicsModel):
    def __init__(self, units=64):
        """Initializes a dynamics model for our door task."""

        # Here, state dim is 5. 3 for x/y/theta, 1 for forward velocity, 1 for angular velocity
        super().__init__(state_dim=5)

        # Control dim is 2, one for forward velocity, one for angular velocity
        # At dynamics update, we take the new input control (velocity) 
        # and replace the old velocity in the state
        control_dim = 2
        self.delta_t = SOMETHING # defined by kitti. 0.1s?

        # Fixed dynamics covariance
        self.Q_scale_tril = nn.Parameter(
            # I think this is effectively saying, we have 0 uncertainty in position dynamics
            # (since we know newton's law applies) and 0.05 uncertainty in velocity dynamics (m/s) 
            torch.cholesky(torch.diag(torch.FloatTensor([0.00, 0.00, 0.00, 0.05, 0.05]))),
            requires_grad=False,
        )

        # Build neural network
        self.state_layers = layers.state_layers(units=units)
        self.control_layers = layers.control_layers(units=units)
        self.shared_layers = nn.Sequential(
            nn.Linear(units * 2, units),
            resblocks.Linear(units),
            resblocks.Linear(units),
            resblocks.Linear(units),
            nn.Linear(units, self.state_dim + 1),
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

        # # (N, control_dim) => (N, units // 2)
        # control_features = self.control_layers(controls)

        # # (N, state_dim) => (N, units // 2)
        # state_features = self.state_layers(initial_states)

        # # (N, units)
        # merged_features = torch.cat((control_features, state_features), dim=-1)

        # # (N, units * 2) => (N, state_dim + 1)
        # output_features = self.shared_layers(merged_features)

        # # We separately compute a direction for our network and a scalar "gate"
        # # These are multiplied to produce our final state output
        # state_update_direction = output_features[..., :state_dim]
        # state_update_gate = torch.sigmoid(output_features[..., -1:])
        # state_update = state_update_direction * state_update_gate

        # Since our control actions are forward/angular velocities, and 
        # our state is x/y/theta/forward velocity/angular velocity, 
        # we can just replace the last two state dimensions with the control actions, 
        # and update the position accordingly (after doing the appropriate transformations) 
        
        # First calculate the rotation matrix to rotate forward/velocity vector into 
        # x velocity, y velocity, and angular velocity
        x, y, theta, forward_v, theta_v = torch.unbind(initial_states, dim=-1)
        
        # Get new velocities from the control inputs 
        forward_v = controls[..., 0]
        theta_v = controls[..., 1]

        # Update theta 
        theta += theta_v * self.delta_t
        
        # Update x and y
        x += forward_v * torch.cos(theta) * self.delta_t
        y += forward_v * torch.sin(theta) * self.delta_t
        
        # Return residual-style state update, constant uncertainties
        states_new = torch.stack((x, y, theta, forward_v, theta_v), dim=-1)
        scale_trils = self.Q_scale_tril[None, :, :].expand(N, state_dim, state_dim)
        return states_new, scale_trils
