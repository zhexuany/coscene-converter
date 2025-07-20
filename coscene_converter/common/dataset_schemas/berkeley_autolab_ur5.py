"""Berkeley Autolab UR5 数据集模式实现"""

from typing import Dict, Any
from foxglove import Channel
from foxglove.schemas import RawImage, FrameTransform, Vector3, Quaternion
from foxglove.channels import RawImageChannel, FrameTransformChannel

from coscene_converter.common.schemas import DatasetSchema, language_instruction_schema, float_schema, joint_state_schema
from coscene_converter.common.dataset_schemas.default import DefaultSchema


class BerkeleyAutolabUR5Schema(DefaultSchema):
    """Berkeley Autolab UR5 数据集模式"""
    
    def setup_channels(self) -> Dict[str, Channel]:
        """设置 Berkeley Autolab UR5 数据集的通道"""
        # 使用默认通道设置
        return super().setup_channels()
    
    def process_step(self, step: Dict[str, Any], channels: Dict[str, Channel]) -> None:
        """处理 Berkeley Autolab UR5 数据集的步骤数据"""
        # 使用默认处理逻辑
        super().process_step(step, channels)