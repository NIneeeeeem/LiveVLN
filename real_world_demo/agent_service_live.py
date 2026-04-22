import numpy as np
import os
import torch
import random
from concurrent.futures import ThreadPoolExecutor
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
import re, cv2
from PIL import Image
from qwen_vl_utils import process_vision_info
from vllm.multimodal.utils import encode_image_base64
import argparse, io
import base64
from flask import Flask, request, jsonify
import time
import json

SYSTEM_PROMPT = "You are a helpful assistant."

app = Flask(__name__)


def decode_rgb_image(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.asarray(image.convert('RGB'))

class NaVid_Agent():
    def __init__(self, model_path, forward_distance, 
                    turn_angle, max_action_history, resolution_ratio, use_video, num_generations = 1, require_map=False):

        self.require_map = require_map
        self.use_video = use_video
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations
        self.pad_action_last_round = None

        model_init_kwargs = {}
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
        model_init_kwargs["use_cache"] = True
        model_init_kwargs['torch_dtype'] = torch.bfloat16
        if "qwen2_vl" in model_path.lower() or "qwen2-vl" in model_path.lower():
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **model_init_kwargs)
        elif "qwen2.5_vl" in model_path.lower() or "qwen2.5-vl" in model_path.lower():
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_init_kwargs)
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_init_kwargs)

        self.device = 'cuda'
        self.model.to(self.device)
        self.model = self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.image_processor.max_pixels = 501760
        print("Initialization Complete")

        self.promt_template = "Imagine you are a robot programmed for navigation tasks. "\
            "You have been given a video of historical observations and an image of the current observation. "\
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
            "which could involve turning left or right by a specific degree or moving forward a certain distance."
        
        self.generation_config = GenerationConfig(
            do_sample=False,  
            temperature=0.2,
            max_new_tokens=512,
            top_p=1.0,
            use_cache=True,
        )
        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})

        self.count_id = 0
        self.reset()

    def uniform_sample_with_ends(self, data, n):
        # n > 2
        if len(data) <= n:
            return data

        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]
    
    def predict_inference(self, prefix_actions=None):
        texts = [self.processor.apply_chat_template( self.conversations, tokenize=False, add_generation_prompt=True)]
        if prefix_actions is not None:
            texts[0] += " " + prefix_actions
            # texts[0] += prefix_actions # TODO
            if not prefix_actions.endswith(","):
                texts[0] += ","
        if self.use_video:
            video_inputs = []
            imgs, vids = process_vision_info(self.conversations)
            video_inputs.append(vids[0])

            prompt_inputs = self.processor(
                text=texts,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            )

        else:
            image_inputs = []
            imgs, vids = process_vision_info(self.conversations)
            image_inputs.append(imgs)

            prompt_inputs = self.processor(
                text=texts,
                images=image_inputs,
                return_tensors="pt",
                padding=True,
            )

        prompt_inputs.to(self.device)
        start_time = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
                use_model_defaults=False,
                )
        print(f"inference time = {time.time() - start_time}")
        # output_ids = outputs.sequences
        output_ids = outputs
        input_token_len = prompt_inputs["input_ids"].shape[1]
        n_diff_input_output = (prompt_inputs["input_ids"] != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs_text = self.processor.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        outputs_text = outputs_text[0]
        outputs_text = outputs_text.strip()
        if prefix_actions is not None:
            outputs_text = prefix_actions + ", " + outputs_text
        return outputs_text

    def extract_multi_result(self, output):
        sub_actions = output.split(', ')
        result = []
        if self.pad_action_last_round is not None:
            self.pad_action_last_round = sub_actions[1] # 第二个动作需要加入队列
        else:
            self.pad_action_last_round = sub_actions[0] # 已有前缀，则第一个动作需要执行
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            result.append([action_index, numeric])
        return result

    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        output_match = re.search(r'<answer>(.*?)</answer>', output)
        output = output_match.group(1).strip() if output_match else output.strip()

        output = output.lower()
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 1, self.forward_distance
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, self.turn_angle
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, self.turn_angle
            match = match.group()
            return 3, float(match)
        return None, None

    def action_id_to_str(self,action_id):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
        if action_id == 0:
            return "stop"
        elif action_id == 1:
            return "forward"
        elif action_id == 2:
            return "turn left"
        elif action_id == 3:
            return "turn right"
        else:
            raise ValueError(f"Invalid action ID: {action_id}")
        
    def reset(self):       

        self.history_rgb_tensor = None
        self.entropy_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id = 0
        self.count_stop = 0
        self.pending_action_list = []
        self.rgb_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        self.pad_action_last_round = None
    
    def preprocess_rgb(self, rgb):
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb, (0, 0), fx=self.resolution_ratio, fy=self.resolution_ratio)
        return Image.fromarray(rgb.astype('uint8')).convert('RGB')

    def split_navigation_actions(self, output):
        output_match = re.search(r'<answer>(.*?)</answer>', output, flags=re.IGNORECASE | re.DOTALL)
        output = output_match.group(1).strip() if output_match else output.strip()

        sub_actions = [
            item.strip()
            for item in re.split(r',|\n|;|\band\b|\bthen\b', output, flags=re.IGNORECASE)
            if item.strip()
        ]
        if not sub_actions and output:
            sub_actions = [output]

        valid_actions = []
        for sub_action in sub_actions:
            action_index, _ = self.extract_result(sub_action)
            if action_index is not None:
                valid_actions.append(sub_action)

        return valid_actions

    def select_action_pair(self, output):
        actions = self.split_navigation_actions(output)
        if len(actions) >= 2:
            return actions[0], actions[1], actions
        if len(actions) == 1:
            return actions[0], "stop", actions + ["stop"]
        return "stop", "stop", ["stop", "stop"]

    def act(self, observations):
        rgb_frames = observations.get("rgb_list")
        if rgb_frames is None:
            rgb_frames = [observations["rgb"]]

        self.rgb_list += [self.preprocess_rgb(rgb) for rgb in rgb_frames]
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[-self.max_action_history:]

        # for observation1+observation2 action style
        self.conversations = self.conversations[:1]
        content = []

        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
        if len(self.rgb_list) > 1:
            if self.use_video:
                content.append({"type": "video", "video": self.uniform_sample_with_ends(self.rgb_list[:-1],8)})
            else:
                # content.extend([{"type": "image_url", "image_url":f"data:image/jpeg;base64,{encode_image_base64(item)}"} for item in self.uniform_sample_with_ends(self.rgb_list[:-1],8)])
                content.extend([{"type": "image", "image": item} for item in self.uniform_sample_with_ends(self.rgb_list[:-1],8)])
        else:
            # content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"})
            content.append({"type": "image", "image": self.rgb_list[-1]})
            # content.extend([{"type": "image", "image": item} for item in self.uniform_sample_with_ends(self.rgb_list,8)])
        content.append({"type": "text", "text": 'and an image of the current observation'})
        # content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"})
        content.append({"type": "image", "image": self.rgb_list[-1]})
        item = self.promt_template.format(observations["instruction"]).split('current observation')
        content.append({"type": "text", "text": item[1]})

        self.conversations.append({
                "role": "user",
                "content": content
            })

        navigation = self.predict_inference()
        self.count_id += 1
        action_e, guard_g, actions = self.select_action_pair(navigation)

        return {
            "navigation": navigation,
            "action_e": action_e,
            "guard_g": guard_g,
            "actions": actions,
        }


@app.route("/get_action",methods=['POST'])
def get_action():
    print("recieve request")
    start = time.time()
    goal = request.form.get('goal')
    text = request.form.get('text', '')
    if goal:
        try:
            text = json.loads(goal)
        except json.JSONDecodeError:
            text = goal
    if not isinstance(text, str):
        text = str(text)

    history_files = [
        file_storage
        for key, file_storage in request.files.items()
        if key.startswith("history_") and key.endswith("_image")
    ]
    image_file = request.files.get('image')
    ordered_files = history_files + ([image_file] if image_file is not None else [])

    if not ordered_files:
        return jsonify({'error': 'no image input found'}), 400

    image_payloads = [file_storage.read() for file_storage in ordered_files]
    if len(image_payloads) == 1:
        image_list = [decode_rgb_image(image_payloads[0])]
    else:
        max_workers = min(len(image_payloads), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            image_list = list(executor.map(decode_rgb_image, image_payloads))

    observation = {
        'rgb_list': image_list,
        'instruction':text
    }
    inference_start = time.time()
    result = agent.act(observation)
    inference_end = time.time()
    print(f"Total processing time: {inference_end - start:.2f} seconds")
    return jsonify({
        'action_e': result['action_e'],
        'guard_g': result['guard_g'],
        'action': [result['action_e'], result['guard_g']],
        'actions': result['actions'],
        'llm_output': result['navigation'],
    })


@app.route("/reset",methods=['POST'])
def reset():
    agent.reset()
    return jsonify({'msg': 'success!'})

if __name__ == '__main__':
    global agent
    agent = NaVid_Agent(model_path='path_to_navida', forward_distance=25, turn_angle=15, max_action_history=200, resolution_ratio=0.5, use_video=False)
    app.run(host='0.0.0.0', port=5802)