import argparse 
import crossmodal.kitti_models.pf
import fannypack 
import torchfilter
import os 
import crossmodal
import torch 
import numpy as np 
import pandas as pd
import pickle

import crossmodal.kitti_models

Task = crossmodal.tasks._kitti.KittiTask

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str) 
parser.add_argument("--checkpoint_label", type=str, default=None)
parser.add_argument

args = parser.parse_args() 

# Create Buddy and read experiment metadata 
buddy = fannypack.utils.Buddy(args.experiment_name)
model_type = buddy.metadata["model_type"] 
dataset_args = buddy.metadata["dataset_args"] 

# Load model using experiment metadata 
# filter_model = crossmodal.kitti_models.pf.KittiParticleFilter(
#     remember_particle_states=True,
#     remember_particle_weights=True
# )

filter_model = Task.model_types[model_type](
    remember_particle_states=True,
    remember_particle_weights=True
)

buddy.attach_model(filter_model) 
buddy.load_checkpoint(label=args.checkpoint_label) 

# Run eval 
eval_helpers = crossmodal.eval_helpers 
eval_helpers.configure(buddy=buddy, task=Task, dataset_args=dataset_args)

# Get eval trajectories 
# Add extras to get the positions for evaluating ground truth 
trajectories, extras = Task.get_eval_trajectories(extras=True, **dataset_args)
assert type(trajectories) == list 

# Put the model in to eval mode and run the trajectory
filter_model.eval() 

with torch.no_grad(): 
    states = [] 
    observations = fannypack.utils.SliceWrapper({}) 
    controls = [] 
    min_timesteps = min([s.shape[0] for s, o, c, in trajectories])
    
    for traj_index, (s, o, c) in enumerate(trajectories): 
        # Update batch 
        states.append(s[:min_timesteps]) 
        observations.append(fannypack.utils.SliceWrapper(o)[:min_timesteps])
        controls.append(c[:min_timesteps])

        device = next(filter_model.parameters()).device
        stack_fn = lambda list_value: fannypack.utils.to_torch(
            np.stack(list_value, axis=1), device=device
        )

        states = stack_fn(states)
        observations = observations.map(stack_fn)
        controls = stack_fn(controls)

        assert states.shape[:2] == controls.shape[:2] 
        assert states.shape[:2] == fannypack.utils.SliceWrapper(observations).shape[:2]
        T, N = states.shape[:2] 
        
        # Initialize belief 
        state_dim = filter_model.state_dim 
        
        cov = (torch.eye(state_dim, device=device) * 0.1)[None, :, :].expand(
            (N, state_dim, state_dim)
        )
        filter_model.initialize_beliefs(
            mean=states[0],
            covariance=cov,
        )
        
        # Normalized predicted states
        predicted_states = filter_model.forward_loop(
            observations=fannypack.utils.SliceWrapper(observations)[1:],
            controls=controls[1:],
        )

        # Instead of using the forward loop, we can use the forward method. This way, we can 
        # Get particle states and weights per time step 
        
        
        # Convert predicted states and true states to cpu 
        predicted_states = predicted_states.cpu().numpy() 
        states = states.cpu().numpy()
        
        # Unnormalized predicted states 
        dataset_stats = Task.dataset_stats
        true_predicted_states = predicted_states * dataset_stats["state_std"] + dataset_stats["state_mean"]
        true_actual_states = states[1:] * dataset_stats["state_std"] + dataset_stats["state_mean"]
        
        # Check if tensor is 3 dimensional, if it is, collapse dimension 1
        if len(true_predicted_states.shape) == 3: 
            true_predicted_states = true_predicted_states.squeeze(1)
            
        if len(true_actual_states.shape) == 3:
            true_actual_states = true_actual_states.squeeze(1)
        
        # Take both states (N, 5) and put them into a single pandas data frame 
        predicted_states_df = pd.DataFrame(true_predicted_states, columns=["vf", "wz"])
        true_states_df = pd.DataFrame(true_actual_states, columns=["vf", "wz"])

        # Load extras in the true_states_df
        true_states_df["x"] = extras[traj_index]["positions"][1:, 0]
        true_states_df["y"] = extras[traj_index]["positions"][1:, 1]
        true_states_df["theta"] = extras[traj_index]["thetas"][1:]

        # Then do simple dynamics sim to get the x and y positions of the predicted velocity values 
        # Use the true_states_df to get the initial x, y, and theta values
        # Use the predicted_states_df to get the predicted velocity values
        # Use the controls to get the delta time values
        x_pred = []
        y_pred = []
        theta_pred = []

        initial_state = true_states_df.iloc[0]
        x_pred.append(initial_state["x"])
        y_pred.append(initial_state["y"])
        theta_pred.append(initial_state["theta"])

        for i in range(1, len(true_states_df)):
            vf = predicted_states_df.iloc[i-1]["vf"]
            wz = predicted_states_df.iloc[i-1]["wz"]
            dt = extras[traj_index]["dts"][i-1]

            x = x_pred[i-1] + vf * np.cos(theta_pred[i-1]) * dt
            y = y_pred[i-1] + vf * np.sin(theta_pred[i-1]) * dt
            theta = theta_pred[i-1] + wz * dt

            # Wrap theta to be between -pi and pi
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            x_pred.append(x)
            y_pred.append(y)
            theta_pred.append(theta)

        # Add the predicted x, y, and theta values to the predicted_states_df
        predicted_states_df["x"] = x_pred
        predicted_states_df["y"] = y_pred
        predicted_states_df["theta"] = theta_pred


        # Also get particle states and weights per time step 
        particle_states = np.array([state.cpu().numpy() for state in filter_model.particle_states_list])
        particle_weights = np.array([weight.cpu().numpy() for weight in filter_model.particle_weights_list])

        # Collapse axis 1
        particle_states = particle_states.squeeze(1)
        particle_weights = particle_weights.squeeze(1)
        
        # Save the data frame to a csv file, located in the same parent directory as this file 
        current_filepath = os.path.dirname(os.path.realpath(__file__))
        
        predicted_states_df.to_csv(os.path.join(current_filepath, "predicted_states.csv"), index=False)
        true_states_df.to_csv(os.path.join(current_filepath, "true_states.csv"), index=False)
        
        # Save weights matrices as pickle object 
        particles_data = {
            "particle_states": particle_states,
            "particle_weights": particle_weights
        }
        with open(os.path.join(current_filepath, "particle_states.pkl"), "wb") as f: 
            pickle.dump(particles_data, f)

        