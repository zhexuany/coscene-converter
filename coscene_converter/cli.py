"""Command-line interface for coscene-converter."""

import argparse
import os
from coscene_converter.open_x_embodiment.data_loader import load_dataset
from coscene_converter.open_x_embodiment.converter import convert_episode, batch_convert_episodes

def main():
    parser = argparse.ArgumentParser(description="Convert Open-X-Embodiment datasets to MCAP format")
    parser.add_argument("--dataset", default="berkeley_autolab_ur5", help="Dataset name")
    parser.add_argument("--episode", type=int, default=1, help="Episode number")
    parser.add_argument("--batch", action="store_true", help="Process multiple episodes")
    parser.add_argument("--start", type=int, default=1, help="Start episode (for batch mode)")
    parser.add_argument("--end", type=int, default=10, help="End episode (for batch mode)")
    parser.add_argument("--output-dir", default="mcap_files", help="Output directory")
    parser.add_argument("--live", action="store_true", help="Show live preview")
    parser.add_argument("--rate", type=float, default=5.0, help="Playback rate in Hz")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_convert_episodes(
            args.dataset, 
            args.start, 
            args.end, 
            args.output_dir
        )
    else:
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
                args.rate, 
                args.live
            )

if __name__ == "__main__":
    main()