
如何利用刻行时空数据平台和 coStudio 对 Open-X-Embodiment 进行可视化```
pip install tensorflow numpy tensorflow-datasets gcsfs
```


# 提取轨迹数据

以下代码借鉴了 [colab 代码](https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb)

代码直接下载了来自伯克利机器人实验室重点 114 条轨迹数据

```
import tensorflow_datasets as tfds

DATASET = "berkeley_autolab_ur5"
TARGET_EPISODE = 40
CONTROL_RATE_HZ = 5  # Depends on the dataset!


def dataset2path(dataset_name):
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{dataset_name}/{version}"


b = tfds.builder_from_directory(builder_dir=dataset2path(DATASET))
ds = b.as_dataset(split="train[{}:{}]".format(TARGET_EPISODE, TARGET_EPISODE + 1))
episode = next(iter(ds))

print("Successfully loaded the dataset: ", DATASET)

assert "steps" in episode, "The dataset does not contain 'steps' key."
print(f"Number of steps in the episode: {len(episode['steps'])}")
```


# Iterating over the episode’s steps (tutorial 2).
We have our episode loaded, let’s now iterate through all the steps of the episode, and let’s make sure the data we find interesting is there. We will write a function print_step_info:

```
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
```
And call it when we are iterating over the steps in our episode:

for i, step in enumerate(episode["steps"]):
    print_step_info(step)
The terminal output we should now see for each step looks like this:

```
Step 113:
  image shape: (480, 640, 3)
  hand_image shape: (480, 640, 3)
  image_with_depth shape: (480, 640, 1)
  natural language instruction: b'pick up the blue cup and put it into the brown cup. '
  Action rotation delta: [0. 0. 0.]
  Action world vector: [0. 0. 0.]
  Robot state: [-2.9432604  -1.2463449   1.6471968  -1.8868321  -1.705304    3.1966798
  0.5433478   0.18657707  0.08173531  0.64937705  0.7561894  -0.0753561
  0.02843411  0.          0.        ]   	
```

If you've made it this far, great! We are now sure we can access our episode data and start streaming it!



```
import tensorflow_datasets as tfds
import foxglove
from foxglove import Channel
from foxglove.schemas import (
	RawImage,
)
from foxglove.channels import RawImageChannel
import time
```

```
server = foxglove.start_server() # We can start it before fetching tfds dataset 
```

```
language_instruction_schema = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
        },
    },
}
language_instruction_chan = Channel(
    topic="/natural_language_instruction", schema=(language_instruction_schema)
)
```

Before we proceed, let’s discuss some terminology:

A schema describes the structure of a message’s contents. Foxglove defines several schemas with visualization support, and users can define custom schemas using supported encodings. 
A channel provides a way to log related messages sharing the same schema. Each channel is identified by a unique topic name. For Foxglove schemas, the SDK offers type-safe channels for logging messages with a known, matching schema
In short, we have created a schema for language instruction that includes a text entry, and we will publish it on the /natural_language_instruction topic. 

Now, let’s modify our initial iteration through the episode steps in the following way:

We will nest it in a while loop, to keep replaying the episode over and over
The while loop will be nested in a try-except block to capture a keyboard interrupt
At the end of our for loop, we will add a sleep instruction to match our control rate so that the episode is replayed at the same frequency as it was captured
We will create the language instruction message and log it on our channel.

```
try:
    #while True:
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

            time.sleep(1 / CONTROL_RATE_HZ)

except KeyboardInterrupt:
    print("Keyboard interrupt received. Will stop the server.")
finally:
    server.stop()
    print("Server stopped.")

```


```
image_chan = RawImageChannel(topic="/image")
```

```
# Publish the image
image_msg = RawImage(
    data=step["observation"]["image"].numpy().tobytes(),
    width=step["observation"]["image"].shape[1],
    height=step["observation"]["image"].shape[0],
    step=step["observation"]["image"].shape[1] * 3,  # Assuming RGB image
    encoding="rgb8",
)
image_chan.log(image_msg)

time.sleep(1 / CONTROL_RATE_HZ)
```
In this snippet, it’s crucial that we correctly assign the step value of the RawImage that calculates the row stride sizes. For an RGB8 image, this means we multiply the width by 3 channels. If we were dealing with a depth image with float32 values with 32FC1 encoding, we would multiply the width by 4 bytes.


```
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

```
Then, we can create the FrameTransformChannel:

```
transform_chan = FrameTransformChannel(topic="/tf")
```


Afterwards, we can craft a FrameTransform message and log it:


```
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
```

The FrameTransform message is simple; we specify two frames and provide translation and rotation from the parent frame to the child frame. Since the measurement units in the dataset are in the metric system and rotations are represented as quaternions, we don’t need to modify the data in any way to make it usable. 

Now, for the plotting of float value, we will create a custom schema:

```
float_schema = {
    "type": "object",
    "properties": {
        "value": {
            "type": "number",
            "format": "float",
        },
    },
}
```
And we will create a Channel object using this schema, just like we did for the natural language instruction at the beginning of this tutorial:
```
gripper_chan = Channel(
    topic="/gripper_state",
    schema=(float_schema)
)
```
To log the gripper status, we can now do the following:

```
# Publish the gripper state
gripper_msg = {"value": float(robot_state[13])}
gripper_chan.log(gripper_msg)
```


```
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

DATASET = "berkeley_autolab_ur5"
TARGET_EPISODE = 341
CONTROL_RATE_HZ = 5  # Depends on the dataset!


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

filename = f"{DATASET}_episode_{TARGET_EPISODE}.mcap"
# writer = foxglove.open_mcap(filename)
server = foxglove.start_server()


b = tfds.builder_from_directory(builder_dir=dataset2path(DATASET))
ds = b.as_dataset(split="train[{}:{}]".format(TARGET_EPISODE, TARGET_EPISODE + 1))
episode = next(iter(ds))

print("Successfully loaded the dataset: ", DATASET)

assert "steps" in episode, "The dataset does not contain 'steps' key."
print(f"Number of steps in the episode: {len(episode['steps'])}")


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
    #while True: # Uncomment this line to run indefinitely
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

except KeyboardInterrupt:
    print("Keyboard interrupt received. Will stop the server.")
finally:
    server.stop()
    print("Server stopped.")
    writer.close()
    print("MCAP file closed.")
```