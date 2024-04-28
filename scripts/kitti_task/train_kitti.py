import argparse 
import datetime 
from typing import cast 

import crossmodal.kitti_models
import crossmodal.kitti_models.pf
from crossmodal.tasks._kitti import KittiTask
import fannypack 
import crossmodal 

Task = KittiTask

# Parse arguments 
parser = argparse.ArgumentParser() 
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--model_type", type=str, required=True, choices=Task.model_types.keys())
Task.add_dataset_arguments(parser)

# Parse args 
args = parser.parse_args() 
model_type = args.model_type
dataset_args = Task.get_dataset_args(args) 

# Load trajectories into memory 
print("Loading trajectories from KITTI dataset...")
train_trajectories = Task.get_train_trajectories()
eval_trajectories = train_trajectories

print(f"Valid model types: {Task.model_types.keys()}")

# Create model, Buddy
filter_model = Task.model_types[model_type]()

# NOTE: This doesn't work right now
if args.sequential_image_rate > 1: 
    filter_model.know_image_blackout = True 
    
buddy = fannypack.utils.Buddy(args.experiment_name, filter_model)
buddy.set_metadata(
    {
        "model_type": model_type, 
        "dataset_args" : dataset_args,
        "train_start_time": datetime.datetime.now().strftime("%b %d, %Y @ %-H:%M:%S"),
        "commit_hash": fannypack.utils.get_git_commit_hash(crossmodal.__file__),
    }
)

# Configure helpers 
train_helpers = crossmodal.train_helpers
train_helpers.configure(buddy=buddy, trajectories=train_trajectories)

eval_helpers = crossmodal.eval_helpers
eval_helpers.configure(buddy=buddy, task=Task, dataset_args=dataset_args)

# Run model-specific training curriculum 
if isinstance(filter_model, crossmodal.kitti_models.pf.KittiParticleFilter): 
    pass_pretrained = False
    # Pre-train dynamics model (single-step)
    if pass_pretrained: 
        try: 
            buddy.load_checkpoint("phase0-dynamics-ss")
        except: 
            train_helpers.train_pf_dynamics_single_step(epochs=5) 
            buddy.save_checkpoint("phase0-dynamics-ss")
    else: 
        train_helpers.train_pf_dynamics_single_step(epochs=5) 
        buddy.save_checkpoint("phase0-dynamics-ss") 
    
    # Pre-train dynamics (recurrent) 
    if pass_pretrained: 
        try: 
            buddy.load_checkpoint("phase1-dynamics-recurrent")
        except: 
            train_helpers.train_pf_dynamics_recurrent(subsequence_length=4, epochs=5) 
            train_helpers.train_pf_dynamics_recurrent(subsequence_length=8, epochs=5) 
            buddy.save_checkpoint("phase1-dynamics-recurrent")
    else: 
        train_helpers.train_pf_dynamics_recurrent(subsequence_length=4, epochs=5) 
        train_helpers.train_pf_dynamics_recurrent(subsequence_length=8, epochs=5) 
        eval_helpers.log_eval()
        buddy.save_checkpoint("phase1-dynamics-recurrent")
    
    # Freeze dynamics 
    fannypack.utils.freeze_module(filter_model.dynamics_model) 
    
    # Pre-train measurement model 
    train_helpers.train_pf_measurement(epochs=10, batch_size=64) 
    eval_helpers.log_eval() 
    buddy.save_checkpoint("pretrained_measurement")
    
    # Train end-to-end 
    train_helpers.train_e2e(subsequence_length=4, epochs=10, batch_size=16) 
    eval_helpers.log_eval() 
    for end_2_end_idx in range(4): 
        print(f"End2End IDX: {end_2_end_idx}")
        train_helpers.train_e2e(subsequence_length=8, epochs=10, batch_size=10) 
        # eval_helpers.log_eval() 
    buddy.save_checkpoint("after_e2e") 
    eval_helpers.log_eval()

# Add training end time
buddy.add_metadata(
    {
        "train_end_time": datetime.datetime.now().strftime("%b %d, %Y @ %-H:%M:%S"),
    }
)
