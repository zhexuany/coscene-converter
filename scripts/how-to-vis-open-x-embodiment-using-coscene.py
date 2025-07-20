import tensorflow_datasets as tfds
import foxglove
from foxglove import Channel
from foxglove.schemas import (
    RawImage,
    FrameTransform,
    Vector3,
    Quaternion,
)
from foxglove.channels import (
    RawImageChannel,
    FrameTransformChannel,
)
import time
import os

DATASET = "berkeley_autolab_ur5"
# Will iterate through all trajectories from 1 to 341
START_EPISODE = 401
END_EPISODE = 401
CONTROL_RATE_HZ = 5  # Depends on the dataset!

# Create mcap_files folder (if it doesn't exist)
MCAP_DIR = "mcap_files"
os.makedirs(MCAP_DIR, exist_ok=True)


def dataset2path(dataset_name):
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{dataset_name}/{version}"


def print_step_info(step):
    print(f"Step {i}:")
    print(f"  image shape: {step['observation']['image'].shape}")
    print(f"  hand_image shape: {step['observation']['hand_image'].shape}")
    print(f"  image_with_depth shape: {step['observation']['image_with_depth'].shape}")
    print(
        f"  natural language instruction: {step['observation']['natural_language_instruction']}"
    )
    print(f"  Action rotation delta: {step['action']['rotation_delta']}")
    print(f"  Action world vector: {step['action']['world_vector']}")
    print(f"  Robot state: {step['observation']['robot_state']}")


language_instruction_schema = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
        },
    },
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

# Start the server
server = foxglove.start_server()

# Create channels
language_instruction_chan = Channel(
    topic="/natural_language_instruction", schema=(language_instruction_schema)
)
image_chan = RawImageChannel(topic="/image")
hand_image_chan = RawImageChannel(topic="/hand_image")
image_with_depth_chan = RawImageChannel(topic="/image_with_depth")
transform_chan = FrameTransformChannel(topic="/tf")
gripper_chan = Channel(
    topic="/gripper_state",
    schema=(float_schema)
)
joint_state_chan = Channel(
    topic="/joint_state",
    schema=(joint_state_schema)
)

try:
    # Iterate through all trajectories from START_EPISODE to END_EPISODE
    for episode_num in range(START_EPISODE, END_EPISODE + 1):
        print(f"Processing episode {episode_num}")
        
        # Create MCAP file for each trajectory, save to mcap_files folder
        filename = os.path.join(MCAP_DIR, f"{DATASET}_episode_{episode_num}.mcap")
        writer = foxglove.open_mcap(filename)
        
        # Load the dataset
        b = tfds.builder_from_directory(builder_dir=dataset2path(DATASET))
        ds = b.as_dataset(split=f"train[{episode_num}:{episode_num + 1}]")
        
        try:
            episode = next(iter(ds))
            print("Successfully loaded the dataset: ", DATASET)
            
            assert "steps" in episode, "The dataset does not contain 'steps' key."
            print(f"Number of steps in the episode: {len(episode['steps'])}")
            
            # Process each step in the trajectory
            for i, step in enumerate(episode["steps"]):
                print_step_info(step)

                # Publish the natural language instruction
                instruction_str = (
                    step["observation"]["natural_language_instruction"]
                    .numpy()
                    .decode("utf-8")
                )
                instruction_msg = {"text": instruction_str}
                language_instruction_chan.log(instruction_msg)

                # Publish the image
                image_msg = RawImage(
                    data=step["observation"]["image"].numpy().tobytes(),
                    width=step["observation"]["image"].shape[1],
                    height=step["observation"]["image"].shape[0],
                    step=step["observation"]["image"].shape[1] * 3,  # Assuming RGB image
                    encoding="rgb8",
                )
                image_chan.log(image_msg)

                # Publish the hand image
                hand_image_msg = RawImage(
                    data=step["observation"]["hand_image"].numpy().tobytes(),
                    width=step["observation"]["hand_image"].shape[1],
                    height=step["observation"]["hand_image"].shape[0],
                    step=step["observation"]["hand_image"].shape[1] * 3,  # Assuming RGB image
                    encoding="rgb8",
                )
                hand_image_chan.log(hand_image_msg)

                # Publish the image with depth
                image_with_depth_msg = RawImage(
                    data=step["observation"]["image_with_depth"].numpy().tobytes(),
                    width=step["observation"]["image_with_depth"].shape[1],
                    height=step["observation"]["image_with_depth"].shape[0],
                    step=step["observation"]["image_with_depth"].shape[1] * 4,  # Assuming 32FC1 image
                    encoding="32FC1",
                )
                image_with_depth_chan.log(image_with_depth_msg)

                # Publish the end-effector transform
                robot_state = step["observation"]["robot_state"].numpy()
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
                transform_chan.log(transform_msg)

                # Publish the gripper state
                gripper_msg = {"value": float(robot_state[13])}
                gripper_chan.log(gripper_msg)

                # Publish the joint state
                joint_state_msg = {
                    "joint0": float(robot_state[0]),
                    "joint1": float(robot_state[1]),
                    "joint2": float(robot_state[2]),
                    "joint3": float(robot_state[3]),
                    "joint4": float(robot_state[4]),
                    "joint5": float(robot_state[5]),
                }
                joint_state_chan.log(joint_state_msg)

                time.sleep(1 / CONTROL_RATE_HZ)
                
            # Close the MCAP file after processing the current trajectory
            writer.close()
            print(f"MCAP file for episode {episode_num} saved to {filename}")
            
        except StopIteration:
            print(f"Episode {episode_num} not found in dataset.")
            continue
        except Exception as e:
            print(f"Error processing episode {episode_num}: {e}")
            if 'writer' in locals():
                writer.close()
            continue

except KeyboardInterrupt:
    print("Keyboard interrupt received. Will stop the server.")
finally:
    server.stop()
    print("Server stopped.")