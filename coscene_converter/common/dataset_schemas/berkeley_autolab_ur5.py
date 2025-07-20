"""Berkeley Autolab UR5 dataset schema implementation"""

from typing import Dict, Any
from foxglove import Channel
from foxglove.schemas import RawImage, FrameTransform, Vector3, Quaternion
from foxglove.channels import RawImageChannel, FrameTransformChannel

from coscene_converter.common.schemas import DatasetSchema, language_instruction_schema, float_schema, joint_state_schema
from coscene_converter.common.dataset_schemas.default import DefaultSchema


class BerkeleyAutolabUR5Schema(DefaultSchema):
    """Berkeley Autolab UR5 dataset schema"""
    
    def setup_channels(self) -> Dict[str, Channel]:
        """Set up channels for Berkeley Autolab UR5 dataset"""
        # Use default channel setup
        return super().setup_channels()
    
    def process_step(self, step: Dict[str, Any], channels: Dict[str, Channel]) -> None:
        """Process step data for Berkeley Autolab UR5 dataset"""
        # Use default processing logic
        super().process_step(step, channels)