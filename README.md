# coScene Converter

Tools for converting robotics datasets to MCAP format for use with coScene and Foxglove.

## Overview

coScene Converter is a Python library that converts robotics datasets, particularly from the Open-X-Embodiment collection, into MCAP format for visualization in coScene. The tool provides a flexible schema-based approach to handle different dataset structures and formats.

## Features

- Conversion of Open-X-Embodiment datasets to MCAP format
- Support for images, depth data, transforms, and robot state
- Integration with coScene for interactive visualization
- Extensible schema system for supporting different dataset formats
- Dataset structure exploration tools

## Installation

```bash
pip install -e .
```

## Usage

### Converting a Dataset

To convert a dataset episode to MCAP format:

```bash
python -m cli --dataset berkeley_autolab_ur5 --episode 1
```

Options:
- `--dataset DATASET`: Dataset name to convert (e.g., berkeley_autolab_ur5, stanford_robocook_converted_externally_to_rlds)
- `--episode EPISODE`: Episode number to convert (default: 1)
- `--batch`: Process multiple episodes in batch mode
- `--start START`: Start episode number for batch mode (default: 1)
- `--end END`: End episode number for batch mode (default: 10)
- `--output-dir OUTPUT_DIR`: Output directory for generated MCAP files (default: mcap_files)
- `--live`: Show live preview during conversion
- `--rate RATE`: Playback rate in Hz for live preview (default: 5.0)
- `--verbose`: Enable verbose output with step information

### Exploring Dataset Structure

To explore the structure of a dataset before conversion:

```bash
python scripts/dataset_structure_explorer.py --dataset stanford_robocook_converted_externally_to_rlds
```

**Important**: The dataset name must exactly match a registered dataset name in the Open-X-Embodiment collection, as these names are used to load datasets from `tensorflow_datasets`. You can verify registered dataset names in the [Open-X-Embodiment Dataset Spreadsheet](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?gid=0) under the "Registered Dataset Name" column.

This will generate a JSON file with the dataset structure that can be used to create a new schema. Since the nature of different datasets varies significantly, it's essential to understand the underlying meaning of each dataset's fields and structure to create an appropriate schema.

## Creating a New Dataset Schema

To add support for a new dataset:

1. Run the dataset structure explorer to understand the dataset format:
   ```bash
   python scripts/dataset_structure_explorer.py --dataset your_dataset_name
   ```
   Remember that `your_dataset_name` must exactly match a registered dataset name in the Open-X-Embodiment collection.

2. Analyze the generated JSON structure carefully to understand:
   - The semantic meaning of each field in the dataset
   - The relationships between different data elements
   - How the dataset represents robot state, sensor data, and actions

3. Create a new schema file in `common/dataset_schemas/your_dataset_name.py`

4. Implement the schema class following the pattern in existing schemas like `berkeley_autolab_ur5.py` or `stanford_robocook_converted_externally_to_rlds.py`

5. Your schema class should:
   - Inherit from `DefaultSchema` or `DatasetSchema`
   - Implement `setup_channels()` to define the channels for your dataset
   - Implement `process_step()` to process each step of data
   - Optionally implement `print_step_info()` for debugging

## Project Structure

- `cli.py`: Command-line interface for the converter
- `common/`: Common utilities and schema definitions
  - `schemas.py`: Base schema classes and common schema definitions
  - `dataset_schemas/`: Dataset-specific schema implementations
- `open_x_embodiment/`: Tools for working with Open-X-Embodiment datasets
  - `data_loader.py`: Functions for loading datasets
  - `converter.py`: Functions for converting datasets to MCAP
- `scripts/`: Utility scripts
  - `dataset_structure_explorer.py`: Tool for exploring dataset structures

## Contributing

To add support for a new dataset:

1. Explore the dataset structure using the explorer tool
2. Create a new schema file based on the existing examples
3. Test your schema with the converter

## License

Apache 2 License