"""Default dataset schema implementation"""

from typing import Dict, Any
from foxglove import Channel
from foxglove.schemas import RawImage, FrameTransform, Vector3, Quaternion
from foxglove.channels import RawImageChannel, FrameTransformChannel

from common.schemas import DatasetSchema, language_instruction_schema, float_schema, joint_state_schema


class DefaultSchema(DatasetSchema):
    """Default dataset schema for standard Open-X-Embodiment datasets"""
    
    def __init__(self):
        """Initialize the schema with a step index counter"""
        self.step_idx = 0
    
    def setup_channels(self) -> Dict[str, Channel]:
        """Set up default channels"""
        """Print information about a step."""
        # This method should be implemented by subclasses
        pass
            
    def print_step_info(self, step: Dict[str, Any], step_index: int) -> None:
        """Print information about a step."""
        # This method should be implemented by subclasses
        pass
    
    def process_step(self, step: Dict[str, Any], channels: Dict[str, Channel], verbose: bool = False) -> None:
        """Process data for a single step"""
        # Print step information if verbose mode is enabled
        if verbose:
            self.print_step_info(step, self.step_idx)
            
        # Increment step index after processing
        self.step_idx += 1
