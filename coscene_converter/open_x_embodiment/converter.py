import foxglove
import time
import os

from coscene_converter.common.schemas import DatasetSchema


def convert_episode(episode, output_file, dataset_name=None, control_rate_hz=5, live_preview=False, verbose=False):
    """Convert an episode to MCAP format and save to file.
    
    Args:
        episode: The episode data
        output_file: Path to save the MCAP file
        dataset_name: Name of the dataset
        control_rate_hz: Conversion rate in Hz
        live_preview: Whether to show live preview
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # If dataset name is not provided, try to extract from output filename
    if dataset_name is None:
        dataset_name = os.path.basename(output_file).split('_episode_')[0]
    
    # Get dataset schema
    schema = DatasetSchema.get_schema_for_dataset(dataset_name)
    print(f"Using schema for dataset {dataset_name}: {schema.__class__.__name__}")
    
    # Create writer or server
    writer = None
    server = None
    
    if live_preview:
        server = foxglove.start_server()
    else:
        writer = foxglove.open_mcap(output_file)
    
    # 使用模式设置通道
    channels = schema.setup_channels()
    
    try:
        for i, step in enumerate(episode["steps"]):
            schema.process_step(step, channels, verbose)
            
            if live_preview:
                time.sleep(1 / control_rate_hz)
    
    except Exception as e:
        print(f"Error during convertion: {e}")
    finally:
        if server:
            server.stop()
            print("Server stopped.")
        if writer:
            writer.close()
            print(f"MCAP file saved to {output_file}")


def batch_convert_episodes(dataset_name, start_episode, end_episode, output_dir="mcap_files", verbose=False):
    """Convert multiple episodes in batch mode.
    
    Args:
        dataset_name: Name of the dataset
        start_episode: Starting episode number
        end_episode: Ending episode number (inclusive)
        output_dir: Directory to save MCAP files
    """
    from coscene_converter.open_x_embodiment.data_loader import load_dataset
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset builder once
    b, _ = load_dataset(dataset_name, start_episode)
    
    # Process each episode
    for episode_num in range(start_episode, end_episode + 1):
        print(f"Processing episode {episode_num}")
        
        # Load episode
        ds = b.as_dataset(split=f"train[{episode_num}:{episode_num + 1}]")
        
        try:
            episode = next(iter(ds))
            
            # Create output filename
            filename = os.path.join(output_dir, f"{dataset_name}_episode_{episode_num}.mcap")
            
            convert_episode(episode, filename, dataset_name=dataset_name, verbose=verbose)
            
        except StopIteration:
            print(f"Episode {episode_num} not found in dataset.")
            continue
        except Exception as e:
            print(f"Error processing episode {episode_num}: {e}")
            continue