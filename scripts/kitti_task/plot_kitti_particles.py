import matplotlib.pyplot as plt 
import numpy as np 
import pickle 
from matplotlib.animation import FuncAnimation
import os 
import pandas as pd 

# Load the data
current_filepath = os.path.dirname(os.path.realpath(__file__))  
particle_data = pickle.load(open(os.path.join(current_filepath, "particle_states.pkl"), "rb"))

# Get the particle states and weights 
particle_states = particle_data["particle_states"]
particle_weights = particle_data["particle_weights"]

# Also get the true states for this 
true_states_df = pd.read_csv(os.path.join(current_filepath, "true_states.csv"))
normalized_true_states_df = true_states_df.copy()

true_states_vf_mean = true_states_df["vf"].mean()
true_states_vf_std = true_states_df["vf"].std()
true_states_vf_abs_max = max(abs(true_states_df["vf"].min()), abs(true_states_df["vf"].max()))

true_states_wz_mean = true_states_df["wz"].mean()
true_states_wz_std = true_states_df["wz"].std()
true_states_wz_min = true_states_df["wz"].min()
true_states_wz_abs_max = max(abs(true_states_df["wz"].min()), abs(true_states_df["wz"].max()))

# Create me a 5x5 grid of plots 
fig, axs = plt.subplots(5, 5, figsize=(10, 10))

# Plot the particles proportional to size of the weights, scale weights 
# to 40-400 for better visualization 
# Flatten the axis to do them all at once 
# Choose a completely random set of sample times
sample_times = np.random.choice(range(len(particle_states)), 25, replace=False)
for i, ax in enumerate(axs.flat): 

    # Get all 300 particle states and weights at sample_time step i 
    particle_states_current = particle_states[sample_times[i]]
    particle_weights_current = particle_weights[sample_times[i]]

    # Scale the weights to be between 40 and 400
    # Exponential scaling 
    particle_weights_current = 40 + 360 * np.exp(particle_weights_current - particle_weights_current.max())
    # Plot the particles
    # plot as red X

    ax.scatter(particle_states_current[:, 0] * true_states_vf_std + true_states_vf_mean, 
               particle_states_current[:, 1] * true_states_wz_std + true_states_wz_mean, 
               s= particle_weights_current,
               c="red", marker="x", alpha=0.5)
    ax.set_title(f"Time step {sample_times[i]}")

    # Plot true state as a green dot 
    ax.scatter(normalized_true_states_df.iloc[sample_times[i]]["vf"], 
               normalized_true_states_df.iloc[sample_times[i]]["wz"], 
                c="green", s=100)
    
    # Set the axis limits
    ax.set_xlim([-1.2 * true_states_vf_abs_max, 1.2 * true_states_vf_abs_max])
    ax.set_ylim([-1.2 * true_states_wz_abs_max, 1.2 * true_states_wz_abs_max])
    
    ax.grid(True)

fig.suptitle("Particle Filter States Over Time (Particle Size Proportional to Weights)")
fig.tight_layout()
fig.savefig(os.path.join(current_filepath, "particle_filter_states.png"))