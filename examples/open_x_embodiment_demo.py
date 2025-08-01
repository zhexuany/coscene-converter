# Copyright 2025 coScene. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from open_x_embodiment.data_loader import load_dataset
from open_x_embodiment.converter import convert_episode

# Load a dataset
dataset_name = "berkeley_autolab_ur5"
episode_num = 1
dataset, episode = load_dataset(dataset_name, episode_num)

# Convert the episode
output_file = f"mcap_files/{dataset_name}_episode_{episode_num}.mcap"
convert_episode(episode, output_file)

print(f"Episode {episode_num} converted and saved to {output_file}")