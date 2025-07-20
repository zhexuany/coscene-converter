# Common Module

This module provides common utilities and schema definitions for the CoScene converter, particularly focused on dataset schema handling for Open-X-Embodiment datasets.

## Overview

The Common module serves as a foundation for dataset conversion in the CoScene converter. It provides:

1. Base schema definitions and interfaces
2. Common schema components for robotics data
3. Dataset-specific schema implementations
4. Dynamic schema loading based on dataset names

## Components

### schemas.py

Contains the base schema definitions and utilities:

- `DatasetSchema`: Abstract base class that defines the interface for all dataset schemas
- Common schema components:
  - `language_instruction_schema`: Schema for natural language instructions
  - `float_schema`: Schema for simple float values
  - `joint_state_schema`: Schema for robot joint states

### dataset_schemas/

Contains implementations of dataset-specific schemas:

- `default.py`: Default schema implementation for standard Open-X-Embodiment datasets
- `berkeley_autolab_ur5.py`: Schema for Berkeley Autolab UR5 dataset
- `berkeley_gnm_cory_hall.py`: Schema for Berkeley GNM Cory Hall dataset

## Usage

### Getting a Schema for a Dataset

```python
from coscene_converter.common.schemas import DatasetSchema

# Get schema for a specific dataset
schema = DatasetSchema.get_schema_for_dataset("berkeley_gnm_cory_hall")

# Set up channels using the schema
channels = schema.setup_channels()

# Process a step using the schema
schema.process_step(step_data, channels)
```

### Creating a New Dataset Schema

To add support for a new dataset:

1. Create a new file in `dataset_schemas/` named after your dataset (e.g., `new_dataset.py`)
2. Define a schema class that inherits from `DatasetSchema` or extends an existing schema
3. Implement the required methods: `setup_channels()` and `process_step()`

Example:

```python
from typing import Dict, Any
from foxglove import Channel
from coscene_converter.common.schemas import DatasetSchema

class NewDatasetSchema(DatasetSchema):
    def setup_channels(self) -> Dict[str, Channel]:
        # Set up channels specific to this dataset
        # ...
        return channels
        
    def process_step(self, step: Dict[str, Any], channels: Dict[str, Channel]) -> None:
        # Process step data for this dataset
        # ...
```

## Schema Discovery

The `DatasetSchema.get_schema_for_dataset()` method automatically discovers and loads the appropriate schema for a given dataset name using the following process:

1. Convert the dataset name to a schema class name (e.g., `berkeley_gnm_cory_hall` → `BerkeleyGnmCoryHallSchema`)
2. Try to find a module specifically named after the dataset
3. If not found, search in all known schema modules
4. Fall back to the default schema if no specific schema is found
    b.metadata