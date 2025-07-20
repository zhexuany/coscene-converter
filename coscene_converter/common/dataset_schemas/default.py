"""默认数据集模式实现"""

from typing import Dict, Any
from foxglove import Channel
from foxglove.schemas import RawImage, FrameTransform, Vector3, Quaternion
from foxglove.channels import RawImageChannel, FrameTransformChannel

from coscene_converter.common.schemas import DatasetSchema, language_instruction_schema, float_schema, joint_state_schema


class DefaultSchema(DatasetSchema):
    """默认数据集模式，适用于标准的Open-X-Embodiment数据集"""
    
    def setup_channels(self) -> Dict[str, Channel]:
        """设置默认通道"""
        language_instruction_chan = Channel(
            topic="/natural_language_instruction", schema=language_instruction_schema
        )
        image_chan = RawImageChannel(topic="/image")
        hand_image_chan = RawImageChannel(topic="/hand_image")
        image_with_depth_chan = RawImageChannel(topic="/image_with_depth")
        transform_chan = FrameTransformChannel(topic="/tf")
        gripper_chan = Channel(
            topic="/gripper_state",
            schema=float_schema
        )
        joint_state_chan = Channel(
            topic="/joint_state",
            schema=joint_state_schema
        )
        
        return {
            "language_instruction": language_instruction_chan,
            "image": image_chan,
            "hand_image": hand_image_chan,
            "image_with_depth": image_with_depth_chan,
            "transform": transform_chan,
            "gripper": gripper_chan,
            "joint_state": joint_state_chan
        }
    
    def process_step(self, step: Dict[str, Any], channels: Dict[str, Channel]) -> None:
        """处理单个步骤的数据"""
        if "observation" not in step:
            print(f"警告：步骤中没有 'observation' 键")
            return
        
        obs = step["observation"]
        
        # 处理自然语言指令
        if "natural_language_instruction" in obs and "language_instruction" in channels:
            try:
                instruction_str = (
                    obs["natural_language_instruction"]
                    .numpy()
                    .decode("utf-8")
                )
                instruction_msg = {"text": instruction_str}
                channels["language_instruction"].log(instruction_msg)
            except Exception as e:
                print(f"处理自然语言指令时出错: {e}")
        
        # 处理图像
        for img_key in ["image", "hand_image", "image_with_depth"]:
            if img_key in obs and img_key in channels:
                try:
                    img_tensor = obs[img_key]
                    
                    # 确定编码和每像素字节数
                    encoding = "rgb8"  # 默认编码
                    bytes_per_pixel = 3  # 默认每像素字节数
                    
                    # 根据图像类型调整编码
                    if img_key == "image_with_depth":
                        encoding = "32FC1"
                        bytes_per_pixel = 4
                    
                    # 创建图像消息
                    img_msg = RawImage(
                        data=img_tensor.numpy().tobytes(),
                        width=img_tensor.shape[1],
                        height=img_tensor.shape[0],
                        step=img_tensor.shape[1] * bytes_per_pixel,
                        encoding=encoding,
                    )
                    
                    # 发布图像
                    channels[img_key].log(img_msg)
                except Exception as e:
                    print(f"处理图像 {img_key} 时出错: {e}")
        
        # 处理机器人状态
        if "robot_state" in obs and "transform" in channels and "gripper" in channels and "joint_state" in channels:
            try:
                robot_state = obs["robot_state"].numpy()
                
                # 确保机器人状态有足够的元素
                if len(robot_state) >= 14:
                    # 发布末端执行器变换
                    transform_msg = FrameTransform(
                        parent_frame_id="robot_base",
                        child_frame_id="end_effector",
                        translation=Vector3(
                            x=float(robot_state[6]),
                            y=float(robot_state[7]),
                            z=float(robot_state[8]),
                        ),
                        rotation=Quaternion(
                            x=float(robot_state[9]),
                            y=float(robot_state[10]),
                            z=float(robot_state[11]),
                            w=float(robot_state[12]),
                        )
                    )
                    channels["transform"].log(transform_msg)
                    
                    # 发布夹持器状态
                    gripper_msg = {"value": float(robot_state[13])}
                    channels["gripper"].log(gripper_msg)
                    
                    # 发布关节状态
                    joint_state_msg = {
                        "joint0": float(robot_state[0]),
                        "joint1": float(robot_state[1]),
                        "joint2": float(robot_state[2]),
                        "joint3": float(robot_state[3]),
                        "joint4": float(robot_state[4]),
                        "joint5": float(robot_state[5]),
                    }
                    channels["joint_state"].log(joint_state_msg)
            except Exception as e:
                print(f"处理机器人状态时出错: {e}")