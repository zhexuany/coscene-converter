"""Berkeley GNM Cory Hall 数据集模式实现"""

from typing import Dict, Any
from foxglove import Channel
from foxglove.schemas import RawImage, FrameTransform, Vector3, Quaternion
from foxglove.channels import RawImageChannel, FrameTransformChannel

from coscene_converter.common.schemas import DatasetSchema, language_instruction_schema, float_schema, joint_state_schema


class BerkeleyGnmCoryHallSchema(DatasetSchema):
    """Berkeley GNM Cory Hall 数据集模式"""
    
    def setup_channels(self) -> Dict[str, Channel]:
        """设置 Berkeley GNM Cory Hall 数据集的通道"""
        language_instruction_chan = Channel(
            topic="/language_instruction", schema=language_instruction_schema
        )
        image_chan = RawImageChannel(topic="/image")
        position_chan = Channel(
            topic="/position",
            schema={
                "type": "object",
                "properties": {
                    "x": {"type": "float"},
                    "y": {"type": "float"}
                }
            }
        )
        state_chan = Channel(
            topic="/state",
            schema={
                "type": "object",
                "properties": {
                    "x": {"type": "float"},
                    "y": {"type": "float"},
                    "z": {"type": "float"}
                }
            }
        )
        yaw_chan = Channel(
            topic="/yaw",
            schema=float_schema
        )
        
        return {
            "language_instruction": language_instruction_chan,
            "image": image_chan,
            "position": position_chan,
            "state": state_chan,
            "yaw": yaw_chan
        }
    
    def process_step(self, step: Dict[str, Any], channels: Dict[str, Channel]) -> None:
        """处理 Berkeley GNM Cory Hall 数据集的步骤数据"""
        if "observation" not in step:
            print(f"警告：步骤中没有 'observation' 键")
            return
        
        obs = step["observation"]
        
        # 处理语言指令
        if "language_instruction" in step and "language_instruction" in channels:
            try:
                instruction_str = (
                    step["language_instruction"]
                    .numpy()
                    .decode("utf-8")
                )
                instruction_msg = {"text": instruction_str}
                channels["language_instruction"].log(instruction_msg)
            except Exception as e:
                print(f"处理语言指令时出错: {e}")
        
        # 处理图像
        if "image" in obs and "image" in channels:
            try:
                img_tensor = obs["image"]
                
                # 创建图像消息
                img_msg = RawImage(
                    data=img_tensor.numpy().tobytes(),
                    width=img_tensor.shape[1],
                    height=img_tensor.shape[0],
                    step=img_tensor.shape[1] * 3,  # RGB 图像
                    encoding="rgb8",
                )
                
                # 发布图像
                channels["image"].log(img_msg)
            except Exception as e:
                print(f"处理图像时出错: {e}")
        
        # 处理位置
        if "position" in obs and "position" in channels:
            try:
                position = obs["position"].numpy()
                position_msg = {
                    "x": float(position[0]),
                    "y": float(position[1])
                }
                channels["position"].log(position_msg)
            except Exception as e:
                print(f"处理位置时出错: {e}")
        
        # 处理状态
        if "state" in obs and "state" in channels:
            try:
                state = obs["state"].numpy()
                state_msg = {
                    "x": float(state[0]),
                    "y": float(state[1]),
                    "z": float(state[2])
                }
                channels["state"].log(state_msg)
            except Exception as e:
                print(f"处理状态时出错: {e}")
        
        # 处理偏航角
        if "yaw" in obs and "yaw" in channels:
            try:
                yaw = obs["yaw"].numpy()
                yaw_msg = {"value": float(yaw[0])}
                channels["yaw"].log(yaw_msg)
            except Exception as e:
                print(f"处理偏航角时出错: {e}")