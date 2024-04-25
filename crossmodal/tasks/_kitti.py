import argparse
import sys
from typing import Any, Dict, List

import fannypack
import numpy as np
import torchfilter
import os 
import pykitti

from ._task import Task


class KittiTask(Task):
    """Dataset definition and model registry for kitti driving task."""

    @classmethod
    def add_dataset_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add dataset options to an argument parser.

        Args:
            parser (argparse.ArgumentParser): Parser to add arguments to.
        """
        parser.add_argument("--sequential_image_rate", type=int, default=1)
        parser.add_argument("--image_blackout_ratio", type=float, default=0.0)
        parser.add_argument("--sequential_gps_rate", type=int, default=1)
        parser.add_argument("--gps_blackout_ratio", type=float, default=0.0)
        parser.add_argument("--full_image_size", action="store_true")

    @classmethod
    def get_dataset_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Get a dataset_args dictionary from a parsed set of arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
        """
        dataset_args = {
            "sequential_image_rate": args.sequential_image_rate,
            "image_blackout_ratio": args.image_blackout_ratio,
            "sequential_gps_rate": args.sequential_gps_rate,
            "gps_blackout_ratio": args.gps_blackout_ratio,
            "full_image_size": args.full_image_size,
        }
        return dataset_args

    @classmethod
    def get_train_trajectories(
        cls, **dataset_args
    ) -> List[torchfilter.types.TrajectoryNumpy]:
        return _load_trajectories(
            "kitti_dataset/2011_09_30/0033", **dataset_args
        )

    @classmethod
    def get_eval_trajectories(
        cls, **dataset_args
    ) -> List[torchfilter.types.TrajectoryNumpy]:
        return _load_trajectories(
            "kitti_dataset/2011_09_30/0033", **dataset_args
        )


def _load_trajectories(
    *input_files,
    sequential_image_rate: int = 1,
    image_blackout_ratio: float = 0.0,
    sequential_gps_rate: int = 1,
    gps_blackout_ratio: float = 0.0,
    start_timestep: int = 0,
    full_image_size: bool = False,
) -> List[torchfilter.types.TrajectoryNumpy]:
    """Loads a list of trajectories from a set of input files, where each trajectory is
    a tuple containing...
        states: an (T, state_dim) array of state vectors
        observations: a key->(T, *) dict of observations
        controls: an (T, control_dim) array of control vectors

    Each input can either be a string or a (string, int) tuple, where int indicates the
    maximum number of trajectories to import.

    Uses pykitti dataset parser for acquiring data. 

    Args:
        *input_files: Trajectory inputs. Should be paths that are in the format 
        /dataset_directory/date/drive_number as specififed by pykitti.
        
        See more here: https://github.com/utiasSTARS/pykitti

    Keyword Args:
        sequential_image_rate (int, optional): If value is `N`, we only send 1 image
            frame ever `N` timesteps. All others are zeroed out.
        image_blackout_ratio (float, optional): Dropout probabiliity for camera inputs.
            0.0 = no dropout, 1.0 = all images dropped out.
        sequential_gps_rate (int, optional): If value is `N`, we only send 1 gps
            reading ever `N` timesteps. All others are zeroed out.
        gps_blackout_ratio (float, optional): Dropout probabiliity for gps inputs.
            0.0 = no dropout, 1.0 = all gps dropped out.
        start_timestep (int, optional): If value is `N`, we skip the first `N` timesteps
            of each trajectory.
        full_image_size (bool, optional): If true, images are not downsampled.

    Returns:
        List[torchfilter.types.TrajectoryNumpy]: list of trajectories.
    """
    trajectories = []

    assert 1 > image_blackout_ratio >= 0
    assert image_blackout_ratio == 0 or sequential_image_rate == 1
    assert 1 > gps_blackout_ratio >= 0
    assert gps_blackout_ratio == 0 or sequential_gps_rate == 1

    for name in input_files:

        # We need to decompose the input files into base_dir, 
        # date, and drive number to use pykitti 
        print(f"From {input_files}")
        drive_path, drive = os.path.split(name) 
        date_path, date = os.path.split(drive_path)
        base_dir = date_path
        
        # Then we need to actually get the kitti trajectory data 
        raw_trajectory = pykitti.raw(base_dir, date, drive)
        
        # Then we need to do some data processing to make sure this is in the right format
        # for our model. We are also going to have one 
        # less timestep than the raw trajectory, since we need the difference image 
        
        start = 1
        timesteps = len(raw_trajectory.timestamps) - start

        # We want: (x, y, theta, forward velocity, angular velocity) 
        # Start by getting the GPS/IMU packets 
        oxts_packets = [packet for packet in raw_trajectory.oxts[start:]]
        
        # Then get positions, we are going to ignore z
        positions = np.array([packet.T_w_imu[:2, 3] for packet in oxts_packets])
        
        # Theta is the yaw angle
        thetas = np.array([packet.packet.yaw for packet in oxts_packets])
        
        # Then get forward and angular velocity 
        forward_velocities = np.expand_dims(np.array([packet.packet.vf for packet in oxts_packets]), 1)
        angular_velocities = np.expand_dims(np.array([packet.packet.wz for packet in oxts_packets]), 1) 
        
        # Then we need to get the images
        all_images = [np.array(image) for image in raw_trajectory.cam2]
        
        # Use difference image to do visual odometry, also include raw image 
        raw_images = np.array(all_images[start:])
        difference_images = np.array([all_images[i] - all_images[i-1] for i in range(start, len(all_images))])
        
        # Optionally, resize these images 
        # You should REALLY do this (i.e. do not disable) 
        # These images are otherwise ginormous 
        if not full_image_size: 
            raw_images = np.array([image[::3, ::3] for image in all_images[start:]])
            difference_images = np.empty_like(raw_images)
            for i in range(1, len(all_images)): 
                np.subtract(all_images[i][::3, ::3], all_images[i-1][::3, ::3], out=difference_images[i-1])
        else: 
            raw_images = np.array(all_images[start:])
            difference_images = np.array([all_images[i] - all_images[i-1] for i in range(start, len(all_images))])
        
        
        # Then put together into states and observations 
        print(f"Data Shapes: {raw_images.shape}, {difference_images.shape}, {positions.shape}, {forward_velocities.shape}, {angular_velocities.shape}")
        states = np.column_stack([positions[:, 0:1], positions[:, 1:2], thetas, forward_velocities, angular_velocities])
        assert states.shape == (timesteps, 5)
        
        observations = {
            "raw_image": raw_images,
            "difference_image": difference_images, 
            "gps_fv": forward_velocities,
            "gps_av": angular_velocities
        }
        
        # Handle sequential and blackout rate 
        if image_blackout_ratio == 0.0: 
            image_mask = np.zeros((timesteps, 1, 1, 1), dtype=np.uint8)
            image_mask[::sequential_image_rate, 0, 0] = 1.0
        else: 
            # Apply blackout rate
            image_mask = (
                (np.random.uniform(size=(timesteps,)) > image_blackout_ratio)
                .astype(np.uint8)
                .reshape((timesteps, 1, 1, 1))
            )
            
        observations["raw_image"] *= image_mask
        observations["difference_image"] *= image_mask
        
        # Do the same for GPS
        if gps_blackout_ratio == 0.0:
            gps_mask = np.zeros((timesteps, 1), dtype=np.float32)
            gps_mask[::sequential_gps_rate, 0,] = 1.0
        else:
            gps_mask = (
                (np.random.uniform(size=(timesteps,)) > gps_blackout_ratio)
                .astype(np.float32)
                .reshape((timesteps, 1))
            )
        
        observations["gps_fv"] *= gps_mask
        observations["gps_av"] *= gps_mask
        
        
        # TODO: Fix controls section, below. Slightly unsure. 
        # Then get controls, which are just the forward and angular velocity
        controls = np.column_stack([forward_velocities, angular_velocities])
        assert controls.shape == (timesteps, 2)
            
        # Finally, normalize everything 
        # States
        states = normalize_data_signal(states) 
        
        # Observations
        observations["raw_image"] = normalize_numpy_images(observations["raw_image"]) 
        observations["difference_image"] = normalize_numpy_images(observations["difference_image"]) 
        observations["gps_fv"] = normalize_data_signal(observations["gps_fv"])
        observations["gps_av"] = normalize_data_signal(observations["gps_av"]) 

        # Controls
        controls = normalize_data_signal(controls)
        
        trajectories.append(
            torchfilter.types.TrajectoryNumpy(
                states[start_timestep:], 
                fannypack.utils.SliceWrapper(observations)[start_timestep:],
                controls[start_timestep:]
            )
        )
        
        del raw_trajectory, all_images, raw_images, difference_images, states, observations, controls

    ## Uncomment this line to generate the lines required to normalize data
    # _print_normalization(trajectories)

    return trajectories

    
def normalize_numpy_images(images): 
    """Normalizes a numpy array of images. Assumes the images are in the range [0, 255]."""
    mean = np.mean(images, axis=(0, 1, 2)) 
    std = np.std(images, axis=(0, 1, 2))
    
    images_normalized = (images - mean)/std
    return images_normalized

def normalize_data_signal(data):
    """Normalizes a numpy array of signals. Assumes the signals are in the range [0, 255]."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    data_normalized = (data - mean)/std
    return data_normalized


def _print_normalization(trajectories):
    """Helper for producing code to normalize inputs"""
    states = []
    observations = fannypack.utils.SliceWrapper({})
    controls = []
    for t in trajectories:
        states.extend(t[0])
        observations.append(t[1])
        controls.extend(t[2])
    observations = observations.map(
        lambda list_value: np.concatenate(list_value, axis=0)
    )
    print("Raw image shape") 
    print(observations["raw_image"].shape)
    print("Difference image shape")
    print(observations["difference_image"].shape)

    def print_ranges(**kwargs):
        for k, v in kwargs.items():
            mean = repr(np.mean(v, axis=0, keepdims=True))
            stddev = repr(np.std(v, axis=0, keepdims=True))
            print(f"{k} -= np.{mean}")
            print(f"{k} /= np.{stddev}")

    print_ranges(
        raw_images=observations["raw_image"],
        difference_images=observations["difference_image"],
        states=states,
        controls=controls,
    )