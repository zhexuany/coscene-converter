from coscene_converter.open_x_embodiment.data_loader import load_dataset
from coscene_converter.open_x_embodiment.converter import convert_episode

# Load a dataset
dataset_name = "berkeley_autolab_ur5"
episode_num = 1
dataset, episode = load_dataset(dataset_name, episode_num)

# Convert the episode
output_file = f"mcap_files/{dataset_name}_episode_{episode_num}.mcap"
convert_episode(episode, output_file)

print(f"Episode {episode_num} converted and saved to {output_file}")