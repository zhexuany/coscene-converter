"""Data loading utilities for Open-X-Embodiment datasets."""

import tensorflow_datasets as tfds
import os

def dataset2path(dataset_name):
    """Convert dataset name to GCS path."""
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{dataset_name}/{version}"

def load_dataset(dataset_name: str, episode_num: int) -> tuple:
    """Load a specific episode from a dataset.
    
    Args:
        dataset_name: Name of the dataset
        episode_num: Episode number to load
        
    Returns:
        tuple: (dataset_builder, episode)
    """
    # Load the dataset
    b = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
    ds = b.as_dataset(split=f"train[{episode_num}:{episode_num + 1}]")
    
    try:
        episode = next(iter(ds))
        print(f"Successfully loaded dataset: {dataset_name}, episode: {episode_num}")
        
        assert "steps" in episode, "The dataset does not contain 'steps' key."
        print(f"Number of steps in the episode: {len(episode['steps'])}")
        
        return b, episode
    except StopIteration:
        print(f"Episode {episode_num} not found in dataset.")
        return b, None
    except Exception as e:
        print(f"Error loading episode {episode_num}: {e}")
        return b, None

def print_step_info(step, step_index):
    """Print information about a step."""
    print(f"Step {step_index}:")
    print(f"  image shape: {step['observation']['image'].shape}")
    print(f"  hand_image shape: {step['observation']['hand_image'].shape}")
    print(f"  image_with_depth shape: {step['observation']['image_with_depth'].shape}")
    print(
        f"  natural language instruction: {step['observation']['natural_language_instruction']}"
    )
    print(f"  Action rotation delta: {step['action']['rotation_delta']}")
    print(f"  Action world vector: {step['action']['world_vector']}")
    print(f"  Robot state: {step['observation']['robot_state']}")