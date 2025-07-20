"""Base schema definitions for dataset processing"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type
from foxglove import Channel
import importlib
import os

# Common schema definitions
language_instruction_schema = {
    "type": "object",
    "properties": {
        "text": {"type": "string"}
    }
}

float_schema = {
    "type": "object",
    "properties": {
        "value": {
            "type": "number",
            "format": "float",
        },
    },
}

joint_state_schema = {
    "type": "object",
    "properties": {
        "joint0": {"type": "number", "format": "float"},
        "joint1": {"type": "number", "format": "float"},
        "joint2": {"type": "number", "format": "float"},
        "joint3": {"type": "number", "format": "float"},
        "joint4": {"type": "number", "format": "float"},
        "joint5": {"type": "number", "format": "float"},
    },
}


class DatasetSchema(ABC):
    """Base class for dataset schemas"""
    
    @abstractmethod
    def setup_channels(self) -> Dict[str, Channel]:
        """Set up channels for the dataset
        
        Returns:
            Dictionary mapping channel names to Channel objects
        """
        pass
    
    @abstractmethod
    def process_step(self, step: Dict[str, Any], channels: Dict[str, Channel], verbose: bool = False) -> None:
        """Process a single step of data
        
        Args:
            step: Step data dictionary
            channels: Dictionary of channels to publish to
            verbose: Whether to print step information
        """
        if verbose:
            self.print_step_info(step, 0) 
        pass
    
    @classmethod
    def get_schema_for_dataset(cls, dataset_name: str) -> 'DatasetSchema':
        """Get the appropriate schema for a dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DatasetSchema instance for the dataset
        """
        # Convert dataset name to schema class name
        parts = dataset_name.split('_')
        class_name = ''.join(part.capitalize() for part in parts) + 'Schema'
        
        print(f"Searching for schema class {class_name}")
        # Try to import the schema class
        try:
            # First try to find a specific schema module
            module_name = f"common.dataset_schemas.{dataset_name}"
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    schema_class = getattr(module, class_name)
                    return schema_class()
            except ImportError:
                # Module not found, try next approach
                pass
            
            # Try to find the class in any of the schema modules
            from common.dataset_schemas import default, berkeley_autolab_ur5, stanford_robocook_converted_externally_to_rlds
            
            for module in [default, berkeley_autolab_ur5, stanford_robocook_converted_externally_to_rlds]:
                if hasattr(module, class_name):
                    schema_class = getattr(module, class_name)
                    return schema_class()
            
            # If not found, use default schema
            print(f"No specific schema found for {dataset_name}, using DefaultSchema")
            from common.dataset_schemas.default import DefaultSchema
            return DefaultSchema()
            
        except Exception as e:
            print(f"Error loading schema for {dataset_name}: {e}")
            from common.dataset_schemas.default import DefaultSchema
            return DefaultSchema()
    
    def print_step_info(self, step: Dict[str, Any], step_index: int) -> None:
        """Print information about a step.
        
        Args:
            step: Step data dictionary
            step_index: Index of the step
        """
        pass