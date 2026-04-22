import numpy as np
import cv2
import ctypes
ctypes.CDLL("/lib/aarch64-linux-gnu/libffi.so.7", mode=ctypes.RTLD_GLOBAL)
import io
import requests
from flask import Flask, jsonify
from PIL import Image
from io import BytesIO
import math, base64
import time, json
import re
from concurrent.futures import ThreadPoolExecutor


from sensor_msgs.msg import Image as ros2Image
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from threading import Lock, Thread
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import logging
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

logger = logging.getLogger(__name__)

class RGBDClient(Node):
    def __init__(self,):
        super().__init__('rgbd_client')
        # config the camera
        self.bridge = CvBridge()
        self.rgbd_group = MutuallyExclusiveCallbackGroup()
        self.rgb_subscription = Subscriber(self, ros2Image, '/camera/captured_image', qos_profile=qos_profile_sensor_data, callback_group=self.rgbd_group)
        self.depth_subscription = Subscriber(self, ros2Image, '/camera/captured_depth', qos_profile=qos_profile_sensor_data, callback_group=self.rgbd_group)
        self.timesynchronizer = ApproximateTimeSynchronizer([self.rgb_subscription, self.depth_subscription], queue_size=10, slop=0.05)
        self.timesynchronizer.registerCallback(self.rgbd_callback)


        logger.info("RGBD Node ready!")
        self.rgb_frame = None
        self.depth_frame = None
        self._lock = Lock()

    def rgbd_callback(self, rgb_msg, depth_msg):
        try:
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            with self._lock:
                self.rgb_frame = rgb_frame
                self.depth_frame = depth_frame
        except Exception as e:
            logger.error(f"Failed to process rgb or depth frame: {str(e)}")
    
    def _capture_loop(self):
        while rclpy.ok():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # scale = 0.001 m/unit, mm->m
                with self._lock:
                    self.rgb_frame = color_image
                    self.depth_frame = depth_image
                self._publish_visualization_rgbd(color_image, depth_image)
                ros_image = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
                ros_depth = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
                current_time = self.get_clock().now().to_msg()
                ros_image.header.stamp = current_time
                ros_depth.header.stamp = current_time
                self.publisher_rgb.publish(ros_image)
                self.publisger_depth.publish(ros_depth)
                

            except Exception as e:
                logger.error(f"Camera error: {e}")
                continue
        self.pipeline.stop()
        logger.info("RealSense pipeline stopped.")
    
    def get_image(self,):
        while rclpy.ok():
            with self._lock:
                rgb_frame = self.rgb_frame
                # depth_frame = self.depth_frame
            if rgb_frame is None:
                logger.error('Got rgb_frame None!')
                return None
            return rgb_frame


# TODO: change in your own machine
action_server = 'http://192.168.112.198:5802/get_action'
reset_server = 'http://192.168.112.198:5802/reset'
rgb_server = "http://127.0.0.1:5000/get_rgbd" 
forward_server = 'http://127.0.0.1:5000/forward'
rotate_server = 'http://127.0.0.1:5000/rotate'

FORWARD_DISTANCE = 0.4
TURN_ANGLE = math.pi / 12
WORD_FORWARD_DISTANCE = 25.0
WORD_TURN_ANGLE = 15.0
MAX_REPEAT_STEPS = 3


def fetch_image():
    rgb = rgbd_node.get_image()
    rgb = rgb[:, :, ::-1]
    rgb = Image.fromarray(rgb)

    return rgb


def encode_observation(rgb_frame):
    # Encode RGB PNG
    color_image = np.asarray(rgb_frame)[:,:,::-1]
    success, color_encoded = cv2.imencode('.png', color_image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise RuntimeError("Failed to encode image")

    color_bytes = io.BytesIO(color_encoded.tobytes())
    color_bytes.seek(0)

    # # Encode depth as .npy
    # depth_bytes = io.BytesIO()
    # np.save(depth_bytes, depth)   # safe binary format
    # depth_bytes.seek(0)

    return color_bytes


def add_observation_files(files, rgb_frame, prefix=None):
    color_bytes = encode_observation(rgb_frame)
    if prefix is None:
        files["image"] = ("color.png", color_bytes, "image/png")
        # files["depth"] = ("depth.npy", depth_bytes, "application/octet-stream")
        return

    files[f"{prefix}_image"] = (f"{prefix}_color.png", color_bytes, "image/png")
    # files[f"{prefix}_depth"] = (f"{prefix}_depth.npy", depth_bytes, "application/octet-stream")


def extract_result(output):
    if output is None:
        return None, None

    if isinstance(output, (int, np.integer)):
        return int(output), None

    if isinstance(output, float) and output.is_integer():
        return int(output), None

    output = str(output).strip()
    if not output:
        return None, None

    output_match = re.search(r'<answer>(.*?)</answer>', output, flags=re.IGNORECASE | re.DOTALL)
    output = output_match.group(1).strip() if output_match else output.strip()
    output = output.lower()

    if "stop" in output:
        return 0, None
    if "forward" in output or "straight" in output:
        match = re.search(r'-?\d+(?:\.\d+)?', output)
        return 1, float(match.group()) if match else WORD_FORWARD_DISTANCE
    if "left" in output:
        match = re.search(r'-?\d+(?:\.\d+)?', output)
        return 2, float(match.group()) if match else WORD_TURN_ANGLE
    if "right" in output:
        match = re.search(r'-?\d+(?:\.\d+)?', output)
        return 3, float(match.group()) if match else WORD_TURN_ANGLE
    return None, None


def words_to_action_ids(action_words):
    if action_words is None:
        return []

    if isinstance(action_words, dict):
        if "action_e" in action_words:
            return words_to_action_ids(action_words["action_e"])
        if "action" in action_words:
            return words_to_action_ids(action_words["action"])
        return []

    if isinstance(action_words, (int, np.integer)):
        return [int(action_words)]

    if isinstance(action_words, float) and action_words.is_integer():
        return [int(action_words)]

    if isinstance(action_words, (list, tuple)):
        action_ids = []
        for item in action_words:
            item_action_ids = words_to_action_ids(item)
            action_ids.extend(item_action_ids)
            if item_action_ids and item_action_ids[-1] == 0:
                break
        return action_ids

    if isinstance(action_words, str):
        stripped = action_words.strip()
        if not stripped:
            return []
        if stripped[0] in "[{" and stripped[-1] in "]}":
            try:
                return words_to_action_ids(json.loads(stripped))
            except json.JSONDecodeError:
                pass

        action_ids = []
        sub_actions = [
            item.strip()
            for item in re.split(r',|\n|;|\band\b|\bthen\b', stripped, flags=re.IGNORECASE)
            if item.strip()
        ]
        if not sub_actions:
            sub_actions = [stripped]

        for sub_action in sub_actions:
            action_index, numeric = extract_result(sub_action)
            if action_index is None:
                continue
            if action_index == 0:
                action_ids.append(0)
                break
            if action_index == 1:
                repeat = max(1, min(MAX_REPEAT_STEPS, round(numeric / WORD_FORWARD_DISTANCE)))
            else:
                repeat = max(1, min(MAX_REPEAT_STEPS, round(numeric / WORD_TURN_ANGLE)))
            action_ids.extend([action_index] * repeat)
        return action_ids

    return []


def split_action_words(action_words):
    if action_words is None:
        return []

    if isinstance(action_words, dict):
        if "action" in action_words:
            return split_action_words(action_words["action"])
        if "action_e" in action_words:
            words = [action_words["action_e"]]
            if "guard_g" in action_words:
                words.append(action_words["guard_g"])
            return words
        return []

    if isinstance(action_words, (int, np.integer)):
        return [int(action_words)]

    if isinstance(action_words, float) and action_words.is_integer():
        return [int(action_words)]

    if isinstance(action_words, (list, tuple)):
        words = []
        for item in action_words:
            item_words = split_action_words(item)
            if item_words:
                words.extend(item_words)
            else:
                words.append(item)
        return words

    output = str(action_words).strip()
    if not output:
        return []

    output_match = re.search(r'<answer>(.*?)</answer>', output, flags=re.IGNORECASE | re.DOTALL)
    output = output_match.group(1).strip() if output_match else output.strip()

    sub_actions = [
        item.strip()
        for item in re.split(r',|\n|;|\band\b|\bthen\b', output, flags=re.IGNORECASE)
        if item.strip()
    ]
    if not sub_actions:
        sub_actions = [output]

    valid_actions = []
    for sub_action in sub_actions:
        action_index, _ = extract_result(sub_action)
        if action_index is not None:
            valid_actions.append(sub_action)

    return valid_actions if valid_actions else [output]


def build_action_request(goal, history_observations=None):
    s = time.time()
    rgb_frame = fetch_image()
    print(time.time()-s)

    files = {}
    add_observation_files(files, rgb_frame)

    history_observations = history_observations or []
    for idx, history_rgb in enumerate(history_observations):
        add_observation_files(files, history_rgb, prefix=f"history_{idx:03d}")

    data = {"goal": json.dumps(goal)}
    if history_observations:
        data["history_frame_count"] = str(len(history_observations))

    return files, data


def request_action(files, data):
    start_time = time.time()
    result = requests.post(action_server,files=files, data = data)
    print(f"time consuming  = {time.time() - start_time}")
    result.raise_for_status()
    result = result.json()
    action = result.get('action_e', result.get('action'))
    guard_g = result.get('guard_g')

    if guard_g is None:
        action_words = split_action_words(action)
        if len(action_words) >= 2:
            action, guard_g = action_words[0], action_words[1]
        elif len(action_words) == 1:
            action, guard_g = action_words[0], "stop"
        else:
            action, guard_g = "stop", "stop"

    return action, guard_g


def get_action(goal, history_observations=None):
    files, data = build_action_request(goal, history_observations)

    return request_action(files, data)

def reset():
    response = requests.post(reset_server)
    response.raise_for_status()


def execute_action_id(action_index):
    if action_index == 1:
        data = {
            'direction':'forward',
            'distance':FORWARD_DISTANCE
        }
        response = requests.post(forward_server, json=data)
    elif action_index == 2:
        data = {
            'direction':'left',
            'theta':TURN_ANGLE
        }
        response = requests.post(rotate_server, json=data)
    elif action_index == 3:
        data = {
            'direction':'right',
            'theta':TURN_ANGLE
        }
        response = requests.post(rotate_server, json=data)
    else:
        return

    response.raise_for_status()


def execute_action(action_words):
    action_ids = words_to_action_ids(action_words)
    history_observations = []

    for action_id in action_ids:
        if action_id == 0:
            break
        execute_action_id(action_id)
        history_observations.append(fetch_image())

    return action_ids, history_observations


def should_stop(action_words):
    action_ids = words_to_action_ids(action_words)
    if not action_ids:
        return False
    return action_ids[0] == 0


def has_no_action(action_words):
    return len(words_to_action_ids(action_words)) == 0

def navigation(text = 'move forward'):
    reset()
    count = 1
    with ThreadPoolExecutor(max_workers=1) as executor:
        # 第一轮先执行 action_e，之后返回的时候 action_e 为前缀，只执行 guard_g
        action_e, guard_g = get_action(text, [])
        action_ids, carry_over_history = execute_action(action_e)
        print(f'bootstrap action_e = {action_e}, guard_g = {guard_g}, action_ids = {action_ids}')

        if action_ids and action_ids[-1] == 0:
            print('!!!!!STOPPED!!!!!!')
            return
        if not action_ids:
            print('failed to parse bootstrap action_e')
            return
        c = time.time()
        while count < 100:
            if should_stop(guard_g):
                print('!!!!!STOPPED BY GUARD!!!!!!')
                break
            if has_no_action(guard_g):
                print('failed to parse guard_g')
                break
            
            files, data = build_action_request(text, carry_over_history)
            next_action_future = executor.submit(request_action, files, data)
            start = time.time()
            
            print("Round time", start-c)
            guard_ids, carry_over_history = execute_action(guard_g)
            c = time.time()
            print(f'current guard_g = {guard_g}, guard_ids = {guard_ids}, excute_time = {c-start}')
            

            if guard_ids and guard_ids[-1] == 0:
                print('!!!!!STOPPED BY GUARD!!!!!!')
                break
            if not guard_ids:
                print('failed to execute guard_g')
                break
            c1 = time.time()
            action_e, guard_g = next_action_future.result()
            c2 = time.time()
            print(f'next action_e = {action_e}, next guard_g = {guard_g}, waiting time={c2-c1}')
            count += 1


if __name__ == '__main__':
    # prompt = 'You are now facing the sofa. Turn left, and as you get close to the refrigerator, walk forward. Turn right in front of the cabinet with the microwave, then continue along the path until you stop at the doorway at the end.'
    
    # prompt = 'Turn right, and when you see the sofa, walk straight ahead, then keep close to the sofa until you stop in front of the cabinet with the microwave.'
    
    # prompt = "Turn right and Leave the bedroom, then walk straight ahead, past the glass door, and stop on the left at the first corner."
    
    # prompt = "Turn right and walk straight out of the lobby. Then turn left at the corner and walk along the corridor until you stop at the first corner."
    
    # prompt = "Turn right, walk straight ahead, enter the first room in front of you, and stop in front of the table."
    
    # prompt = "Turn left and go forward and turn right at the first intersection, then keep going forward until you enter the room at the end."
    
    prompt = "go forward to enter the room"
    global rgbd_node 
    rclpy.init() 
    rgbd_node = RGBDClient()
    executor_thread = Thread(target=rclpy.spin, args=(rgbd_node,), daemon=True)
    executor_thread.start()
    print("RGB Thread begin")
    time.sleep(1)
    navigation(prompt)

    action = get_action([1,2])
    print(action)