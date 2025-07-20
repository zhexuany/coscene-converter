"""Command-line interface for coscene-converter."""
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

import argparse
import os
from open_x_embodiment.data_loader import load_dataset
from open_x_embodiment.converter import convert_episode, batch_convert_episodes

def main():
    parser = argparse.ArgumentParser(
        description="Convert Open-X-Embodiment datasets to MCAP format for visualization in coScene",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset", 
        default="berkeley_autolab_ur5", 
        help="Dataset name to convert (e.g., berkeley_autolab_ur5, stanford_robocook_converted_externally_to_rlds)"
    )
    parser.add_argument(
        "--episode", 
        type=int, 
        default=1, 
        help="Episode number to convert"
    )
    parser.add_argument(
        "--batch", 
        action="store_true", 
        help="Process multiple episodes in batch mode"
    )
    parser.add_argument(
        "--start", 
        type=int, 
        default=1, 
        help="Start episode number (for batch mode)"
    )
    parser.add_argument(
        "--end", 
        type=int, 
        default=10, 
        help="End episode number (for batch mode)"
    )
    parser.add_argument(
        "--output-dir", 
        default="mcap_files", 
        help="Output directory for generated MCAP files"
    )
    parser.add_argument(
        "--live", 
        action="store_true", 
        help="Show live preview during conversion"
    )
    parser.add_argument(
        "--rate", 
        type=float, 
        default=5.0, 
        help="Playback rate in Hz for live preview"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output with step information"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.batch:
        print(f"Batch converting episodes {args.start} to {args.end} from dataset '{args.dataset}'")
        batch_convert_episodes(
            args.dataset, 
            args.start, 
            args.end, 
            args.output_dir
        )
        print(f"Batch conversion complete. Output files saved to {args.output_dir}/")
    else:
        print(f"Converting episode {args.episode} from dataset '{args.dataset}'")
        # Load dataset
        _, episode = load_dataset(args.dataset, args.episode)
        
        if episode is not None:
            # Create output filename
            filename = os.path.join(
                args.output_dir, 
                f"{args.dataset}_episode_{args.episode}.mcap"
            )
            
            # Convert
            convert_episode(
                episode, 
                filename, 
                args.dataset,  
                args.rate, 
                args.live,
                verbose=args.verbose
            )
            print(f"Conversion complete. Output saved to {filename}")
        else:
            print(f"Error: Could not load episode {args.episode} from dataset '{args.dataset}'")

if __name__ == "__main__":
    main()