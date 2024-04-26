import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 

# Get the path of the current file 
current_filepath = os.path.dirname(os.path.realpath(__file__))

# Load the predicted and true states
predicted_states = pd.read_csv(os.path.join(current_filepath, "predicted_states.csv"))
true_states = pd.read_csv(os.path.join(current_filepath, "true_states.csv"))

# Plot the predicted and true states, x on x axis, y on y axis 
plt.figure()
plt.plot(predicted_states["x"], predicted_states["y"], label="Predicted States")
plt.plot(true_states["x"], true_states["y"], label="True States")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

# Save the plot
plt.savefig(os.path.join(current_filepath, "predicted_vs_true_states.png"))

# Then also create theta vs time plot, fv vs time plot, wz vs time plot 
plt.figure()
plt.plot(np.rad2deg(predicted_states["theta"]), label="Predicted Theta")
plt.plot(np.rad2deg(true_states["theta"]), label="True Theta")
plt.xlabel("Timestep")
plt.ylabel("Theta")
plt.legend()
plt.grid()

plt.savefig(os.path.join(current_filepath, "predicted_vs_true_theta.png"))

plt.figure()
plt.plot(predicted_states["vf"], label="Predicted Forward Velocity")
plt.plot(true_states["vf"], label="True Forward Velocity")
plt.xlabel("Timestep")
plt.grid()

plt.savefig(os.path.join(current_filepath, "predicted_vs_true_vf.png"))

plt.figure()
plt.plot(np.rad2deg(predicted_states["wz"]), label="Predicted Angular Velocity")
plt.plot(np.rad2deg(true_states["wz"]), label="True Angular Velocity")
plt.xlabel("Timestep")
plt.grid()

plt.savefig(os.path.join(current_filepath, "predicted_vs_true_wz.png"))

