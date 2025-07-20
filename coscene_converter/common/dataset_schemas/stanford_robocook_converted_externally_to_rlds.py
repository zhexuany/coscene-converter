"""Stanford RoboCook dataset schema implementation"""

from typing import Dict, Any
from foxglove import Channel
from foxglove.schemas import RawImage, FrameTransform, Vector3, Quaternion
from foxglove.channels import RawImageChannel, FrameTransformChannel

from coscene_converter.common.schemas import DatasetSchema, language_instruction_schema, float_schema, joint_state_schema


class StanfordRobocookConvertedExternallyToRldsSchema(DatasetSchema):
    """Stanford RoboCook dataset schema"""
    
    def setup_channels(self) -> Dict[str, Channel]:
        """Set up channels for Stanford RoboCook dataset"""
        # Language instruction channel
        language_instruction_chan = Channel(
            topic="/language_instruction", schema=language_instruction_schema
        )
        
        # Image channels (4 cameras)
        image_1_chan = RawImageChannel(topic="/image_1")
        image_2_chan = RawImageChannel(topic="/image_2")
        image_3_chan = RawImageChannel(topic="/image_3")
        image_4_chan = RawImageChannel(topic="/image_4")
        
        # Depth channels (4 cameras)
        depth_1_chan = RawImageChannel(topic="/depth_1")
        depth_2_chan = RawImageChannel(topic="/depth_2")
        depth_3_chan = RawImageChannel(topic="/depth_3")
        depth_4_chan = RawImageChannel(topic="/depth_4")
        
        # Robot state channels
        joint_state_chan = Channel(
            topic="/joint_state",
            schema=joint_state_schema
        )
        
        return {
            "language_instruction": language_instruction_chan,
            "image_1": image_1_chan,
            "image_2": image_2_chan,
            "image_3": image_3_chan,
            "image_4": image_4_chan,
            "depth_1": depth_1_chan,
            "depth_2": depth_2_chan,
            "depth_3": depth_3_chan,
            "depth_4": depth_4_chan,
            "joint_state": joint_state_chan
        }
    
    def print_step_info(self, step: Dict[str, Any], step_index: int) -> None:
        """Print information about a step for Stanford RoboCook dataset"""
        print(f"Step {step_index}:")
        
        # Print language instruction
        if "language_instruction" in step:
            try:
                instruction_str = step["language_instruction"].numpy().decode("utf-8")
                print(f"  Language instruction: {instruction_str}")
            except Exception as e:
                print(f"  Error processing language instruction: {e}")
        
        # Print image information
        if "observation" in step:
            obs = step["observation"]
            
            # Print camera images information
            for i in range(1, 5):
                img_key = f"image_{i}"
                if img_key in obs:
                    try:
                        print(f"  {img_key} shape: {obs[img_key].shape}")
                    except Exception as e:
                        print(f"  Error accessing {img_key}: {e}")
            
            # Print depth images information
            for i in range(1, 5):
                depth_key = f"depth_{i}"
                if depth_key in obs:
                    try:
                        print(f"  {depth_key} shape: {obs[depth_key].shape}")
                    except Exception as e:
                        print(f"  Error accessing {depth_key}: {e}")
            
            # Print robot state
            if "state" in obs:
                try:
                    state_tensor = obs["state"].numpy()
                    print(f"  Robot state: {state_tensor}")
                    if len(state_tensor) >= 7:
                        print(f"    Joint states: {state_tensor[:6]}")
                except Exception as e:
                    print(f"  Error processing robot state: {e}")
    
    def process_step(self, step: Dict[str, Any], channels: Dict[str, Channel], verbose: bool = False) -> None:
        """Process a single step of data for Stanford RoboCook dataset"""
        # Print step information if verbose mode is enabled
        if verbose:
            self.print_step_info(step, 0)  # Step index not available here, using 0
        
        # Process language instruction
        if "language_instruction" in step and "language_instruction" in channels:
            try:
                instruction_str = step["language_instruction"].numpy().decode("utf-8")
                instruction_msg = {"text": instruction_str}
                channels["language_instruction"].log(instruction_msg)
            except Exception as e:
                print(f"Error processing language instruction: {e}")
        
        # Process images
        if "observation" in step:
            obs = step["observation"]
            
            # Process all camera images
            for i in range(1, 5):
                img_key = f"image_{i}"
                if img_key in obs and img_key in channels:
                    try:
                        img_tensor = obs[img_key]
                        img_msg = RawImage(
                            data=img_tensor.numpy().tobytes(),
                            width=img_tensor.shape[1],
                            height=img_tensor.shape[0],
                            step=img_tensor.shape[1] * 3,  # RGB image (3 bytes per pixel)
                            encoding="rgb8",
                        )
                        channels[img_key].log(img_msg)
                    except Exception as e:
                        print(f"Error processing {img_key}: {e}")
            
            # Process all depth images
            for i in range(1, 5):
                depth_key = f"depth_{i}"
                if depth_key in obs and depth_key in channels:
                    try:
                        depth_tensor = obs[depth_key]
                        depth_msg = RawImage(
                            data=depth_tensor.numpy().tobytes(),
                            width=depth_tensor.shape[1],
                            height=depth_tensor.shape[0],
                            step=depth_tensor.shape[1] * 4,  # Float32 depth (4 bytes per pixel)
                            encoding="32FC1",
                        )
                        channels[depth_key].log(depth_msg)
                    except Exception as e:
                        print(f"Error processing {depth_key}: {e}")
            
            # Process robot state
            if "state" in obs and "joint_state" in channels:
                try:
                    state_tensor = obs["state"].numpy()
                    # Ensure we have enough elements (6 DOF robot + 1 gripper)
                    if len(state_tensor) >= 7: 
                        # Publish joint state
                        joint_state_msg = {
                            "joint0": float(state_tensor[0]),
                            "joint1": float(state_tensor[1]),
                            "joint2": float(state_tensor[2]),
                            "joint3": float(state_tensor[3]),
                            "joint4": float(state_tensor[4]),
                            "joint5": float(state_tensor[5])
                        }
                        channels["joint_state"].log(joint_state_msg)
                except Exception as e:
                    print(f"Error processing robot state: {e}")