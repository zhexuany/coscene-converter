"""Default dataset schema implementation"""

from typing import Dict, Any
from foxglove import Channel
from foxglove.schemas import RawImage, FrameTransform, Vector3, Quaternion
from foxglove.channels import RawImageChannel, FrameTransformChannel

from coscene_converter.common.schemas import DatasetSchema, language_instruction_schema, float_schema, joint_state_schema


class DefaultSchema(DatasetSchema):
    """Default dataset schema for standard Open-X-Embodiment datasets"""
    
    def setup_channels(self) -> Dict[str, Channel]:
        """Set up default channels"""
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
    
    def print_step_info(self, step: Dict[str, Any], step_index: int) -> None:
        """Print information about a step."""
        print(f"Step {step_index}:")
        if "observation" in step:
            obs = step["observation"]
            if "image" in obs:
                print(f"  image shape: {obs['image'].shape}")
            if "hand_image" in obs:
                print(f"  hand_image shape: {obs['hand_image'].shape}")
            if "image_with_depth" in obs:
                print(f"  image_with_depth shape: {obs['image_with_depth'].shape}")
            if "natural_language_instruction" in obs:
                print(f"  natural language instruction: {obs['natural_language_instruction']}")
            if "robot_state" in obs:
                print(f"  Robot state: {obs['robot_state']}")
        
        if "action" in step:
            action = step["action"]
            if "rotation_delta" in action:
                print(f"  Action rotation delta: {action['rotation_delta']}")
            if "world_vector" in action:
                print(f"  Action world vector: {action['world_vector']}")
        
        # Process natural language instruction
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
                print(f"Error processing natural language instruction: {e}")
        
        # Process images
        for img_key in ["image", "hand_image", "image_with_depth"]:
            if img_key in obs and img_key in channels:
                try:
                    img_tensor = obs[img_key]
                    
                    # Determine encoding and bytes per pixel
                    encoding = "rgb8"  # Default encoding
                    bytes_per_pixel = 3  # Default bytes per pixel
                    
                    # Adjust encoding based on image type
                    if img_key == "image_with_depth":
                        encoding = "32FC1"
                        bytes_per_pixel = 4
                    
                    # Create image message
                    img_msg = RawImage(
                        data=img_tensor.numpy().tobytes(),
                        width=img_tensor.shape[1],
                        height=img_tensor.shape[0],
                        step=img_tensor.shape[1] * bytes_per_pixel,
                        encoding=encoding,
                    )
                    
                    # Publish image
                    channels[img_key].log(img_msg)
                except Exception as e:
                    print(f"Error processing image {img_key}: {e}")
        
        # Process robot state
        if "robot_state" in obs and "transform" in channels and "gripper" in channels and "joint_state" in channels:
            try:
                robot_state = obs["robot_state"].numpy()
                
                # Ensure robot state has enough elements
                if len(robot_state) >= 14:
                    # Publish end effector transform
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
                    
                    # Publish gripper state
                    gripper_msg = {"value": float(robot_state[13])}
                    channels["gripper"].log(gripper_msg)
                    
                    # Publish joint state
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
                print(f"Error processing robot state: {e}")
    
    def process_step(self, step: Dict[str, Any], channels: Dict[str, Channel], verbose: bool = False) -> None:
        """Process data for a single step"""
        if verbose:
            self.print_step_info(step, 0)  # 步骤索引在这里不可用，使用0
            
        if "observation" not in step:
            print(f"Warning: 'observation' key not found in step")
            return
        
        if "observation" not in step:
            print(f"Warning: 'observation' key not found in step")
            return
        
        if "observation" not in step:
            print(f"Warning: 'observation' key not found in step")
            return