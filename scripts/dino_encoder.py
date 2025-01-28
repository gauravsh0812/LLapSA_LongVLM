import os
import math
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, Dinov2Model, Dinov2Config
    

def load_video(vis_path, num_frm=60):
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--video_dir_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path", required=True, help="The output dir to save the features in.")
    # parser.add_argument("--xy", required=True)
    args = parser.parse_args()
    return args


def load_and_stack_hidden_states(temp, video_id, 
                                 counter, vcgpt_features):

    # Iterate over each video ID
    hidden_states = []
    for i in range(counter):
        with open(os.path.join(temp, f"vcgpt_{video_id}_{i}.pkl"), 'rb') as f:
            hidden_state = pickle.load(f)
            hidden_states.append(hidden_state)

        stacked_states = torch.stack(hidden_states, dim=0)
        output_file = os.path.join(vcgpt_features, f"{video_id}.pkl")
        
        with open(output_file, 'wb') as f:
            pickle.dump(stacked_states, f)

class DinoFeatureExtractor:
    def __init__(self, model_name="facebook/dinov2-giant", device="cuda:0"):
        """
        Initializes the DINOv2 model and image processor for feature extraction.
        Args:
            model_name (str): Pre-trained DINOv2 model to use.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        configuration = Dinov2Config.from_pretrained(model_name)
        print(configuration)
        configuration.hidden_size = 1024
        self.processor = AutoImageProcessor.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model = Dinov2Model(configuration).to(self.device)
        self.model.eval()

    def extract_features(self, frames, layer_index=-2):
        """
        Extract features from a batch of video frames at a specific encoder layer.
        
        Args:
            frames (torch.Tensor): A batch of preprocessed video frames, shape (batch_size, channels, height, width).
            layer_index (int): Encoder layer index to extract features from.
        
        Returns:
            torch.Tensor: Extracted features from the specified encoder layer.
        """
        with torch.no_grad():
            # Forward pass through the model
            outputs = self.model(frames, output_hidden_states=True)
            # Extract features from the desired layer
            features = outputs.hidden_states[layer_index]
        return features

    def preprocess_frames(self, frames):
        """
        Preprocesses video frames for input to the DINOv2 model.
        
        Args:
            frames (list or np.ndarray): List of frames or NumPy array of shape (batch_size, height, width, channels).
        
        Returns:
            torch.Tensor: Preprocessed frames as a tensor, shape (batch_size, channels, height, width).
        """
        # Convert frames to PIL Images and preprocess
        inputs = self.processor(frames, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

def main():

    args = parse_args()
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    vcgpt_features = os.path.join(clip_feat_path, "dino_features")
    os.makedirs(vcgpt_features, exist_ok=True)

    # Initialize the DinoV2 model    
    all_videos = [i for i in os.listdir(video_dir_path) if "_60sec_" in i][:1000]
    
    dino = DinoFeatureExtractor()

    video_clip_features = {}
    counter = 0

    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]
        if os.path.exists(f"{vcgpt_features}/{video_id}.pkl"):
            continue
        try:
            frames = load_video(video_path)
            preprocessed_frames = dino.preprocess_frames(frames)
            features = dino.extract_features(preprocessed_frames, layer_index=-2)
            video_clip_features[video_id] = features.half()
            counter += 1       
            
        except Exception as e:
            print(f"Can't process {video_path} due to {e}")
    
        if counter % 50==0:
            for key in video_clip_features.keys():
                features = video_clip_features[key]
                with open(f"{vcgpt_features}/{key}.pkl", 'wb') as f:
                    pickle.dump(features, f)
            video_clip_features = {}
    
    for key in video_clip_features.keys():
        features = video_clip_features[key]
        with open(f"{vcgpt_features}/{key}.pkl", 'wb') as f:
            pickle.dump(features, f)

if __name__ == "__main__":
    main()  


# git pull; CUDA_VISIBLE_DEVICES=0 python scripts/save_vcgpt_features.py --video_dir_path /data/shared/gauravs/llapsa/vcgpt_clips --clip_feat_path /data/shared/gauravs/llapsa/longvu_videos