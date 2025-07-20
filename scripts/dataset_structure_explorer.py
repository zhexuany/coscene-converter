import tensorflow_datasets as tfds
import tensorflow as tf
import argparse
import json

def dataset2path(dataset_name):
    """Convert dataset name to GCS path."""
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{dataset_name}/{version}"

def explore_tensor_structure(tensor, prefix="", max_depth=5, current_depth=0):
    """Recursively explore the structure of a tensor"""
    if current_depth >= max_depth:
        return {"max_depth_reached": True}
    
    if isinstance(tensor, dict):
        result = {}
        for k, v in tensor.items():
            result[k] = explore_tensor_structure(v, prefix + "  ", max_depth, current_depth + 1)
        return result
    elif isinstance(tensor, (list, tuple)) and len(tensor) > 0:
        return {
            "type": f"{type(tensor).__name__}",
            "length": len(tensor),
            "sample": explore_tensor_structure(tensor[0], prefix + "  ", max_depth, current_depth + 1)
        }
    elif isinstance(tensor, tf.Tensor):
        return {
            "type": "tf.Tensor",
            "shape": tensor.shape.as_list(),
            "dtype": tensor.dtype.name
        }
    else:
        return {"type": f"{type(tensor).__name__}"}

def main():
    parser = argparse.ArgumentParser(description="Explore the structure of Open-X-Embodiment datasets")
    parser.add_argument("--dataset", type=str, default="berkeley_autolab_ur5", help="Dataset name")
    parser.add_argument("--episode", type=int, default=1, help="Episode number to load")
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset}, episode: {args.episode}")
    
    # Create a result JSON structure
    result_json = {
        "dataset_name": args.dataset,
        "episode_number": args.episode,
        "episode_structure": {}
    }
    
    # Load the dataset
    try:
        b = tfds.builder_from_directory(builder_dir=dataset2path(args.dataset))
        ds = b.as_dataset(split=f"train[{args.episode}:{args.episode + 1}]")
        episode = next(iter(ds))
        
        print(f"Successfully loaded dataset: {args.dataset}")
        result_json["load_status"] = "success"
        
        # Check dataset structure
        if "steps" in episode:
            steps_count = len(episode['steps'])
            print(f"Number of steps in the episode: {steps_count}")
            result_json["steps_count"] = steps_count
            
            # Explore the structure of the entire episode
            print("\n===== Top-level keys in the Episode =====")
            top_level_keys = list(episode.keys())
            for key in top_level_keys:
                print(f"- {key}")
            result_json["top_level_keys"] = top_level_keys
            
            # Explore the structure of steps - modify this part to handle different types of steps
            print("\n===== Structure of Steps =====")
            steps = episode["steps"]
            steps_type = str(type(steps))
            print(f"Type of Steps: {steps_type}")
            result_json["steps_type"] = steps_type
            
            # Handle different types of steps
            if isinstance(steps, (list, tuple)) and len(steps) > 0:
                # If steps is a list or tuple, directly access the first element
                print("\n===== Top-level keys in the first Step =====")
                step = steps[0]
                step_keys = list(step.keys())
                for key in step_keys:
                    print(f"- {key}")
                result_json["first_step_keys"] = step_keys
                
                # Detailed exploration of the first step's structure
                print("\n===== Detailed structure of the first Step =====")
                structure = explore_tensor_structure(step)
                print(json.dumps(structure, indent=2))
                result_json["first_step_structure"] = structure
                
            elif hasattr(steps, "take") and callable(getattr(steps, "take", None)):
                # If steps is a dataset object, use the take method to get the first element
                print("\n===== Steps is a dataset object =====")
                result_json["steps_is_dataset"] = True
                try:
                    first_step = next(iter(steps.take(1)))
                    print("\n===== Top-level keys in the first Step =====")
                    step_keys = list(first_step.keys())
                    for key in step_keys:
                        print(f"- {key}")
                    result_json["first_step_keys"] = step_keys
                    
                    # Detailed exploration of the first step's structure
                    print("\n===== Detailed structure of the first Step =====")
                    structure = explore_tensor_structure(first_step)
                    print(json.dumps(structure, indent=2))
                    result_json["first_step_structure"] = structure
                except Exception as e:
                    error_msg = str(e)
                    print(f"Cannot access the first step: {error_msg}")
                    result_json["first_step_error"] = error_msg
                    
                    print("Trying to print basic information about steps:")
                    steps_info = {"type": str(type(steps))}
                    print(f"  - Type: {steps_info['type']}")
                    
                    if hasattr(steps, "element_spec"):
                        element_spec = str(steps.element_spec)
                        print(f"  - Element specification: {element_spec}")
                        steps_info["element_spec"] = element_spec
                    
                    result_json["steps_info"] = steps_info
            else:
                # If steps is another type, try to print its basic information
                print(f"Steps is a type that doesn't support direct indexing: {type(steps)}")
                result_json["steps_indexable"] = False
                
                print("Trying to print basic information about steps:")
                steps_info = {}
                if hasattr(steps, "__dict__"):
                    for attr_name in dir(steps):
                        if not attr_name.startswith("_") and not callable(getattr(steps, attr_name)):
                            try:
                                attr_value = getattr(steps, attr_name)
                                attr_value_str = str(attr_value)
                                print(f"  - {attr_name}: {attr_value_str}")
                                steps_info[attr_name] = attr_value_str
                            except:
                                pass
                result_json["steps_info"] = steps_info
            
            # Save the complete episode structure
            result_json["episode_structure"] = explore_tensor_structure(episode)
            
            # Save results to file
            output_file = f"{args.dataset}_structure.json"
            with open(output_file, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"\nStructure has been saved to file: {output_file}")
            
        else:
            print("Warning: Dataset does not have a 'steps' key")
            result_json["warning"] = "Dataset does not have a 'steps' key"
            
            print("Available top-level keys:")
            top_level_keys = list(episode.keys())
            for key in top_level_keys:
                print(f"- {key}")
            result_json["top_level_keys"] = top_level_keys
            
            # Save results to file
            output_file = f"{args.dataset}_structure.json"
            with open(output_file, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"\nStructure has been saved to file: {output_file}")
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        result_json["error"] = error_msg
        
        # Save collected information even if an error occurs
        output_file = f"{args.dataset}_structure.json"
        with open(output_file, "w") as f:
            json.dump(result_json, f, indent=2)
        print(f"\nPartial structure has been saved to file: {output_file}")

if __name__ == "__main__":
    main()