import os 
import numpy as np
import math
import random
import torch
import tqdm
import pickle
import argparse
import multiprocessing as mp
import torch.nn.functional as F
from longvlm.model.merge import merge_tokens
from scripts.dino_encoder import load_video
from transformers import CLIPVisionModel, CLIPImageProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--xy", required=True)
    parser.add_argument("--dino_path", required=True, help="Path to read the dino tensors from.")
    parser.add_argument("--video_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--local_feature_path", required=True, help="path to save local features")
    args = parser.parse_args()
    return args

def reduce_similar_frames(visual_emb_frame):
    
    "https://github.com/Vision-CAIR/LongVU/blob/1ca42869fd456ecfef8acdc2aaa01e43864431e0/longvu/cambrian_arch.py#L1474"
    
    window_size = 5
    assert visual_emb_frame.shape[0] % window_size == 0, "num frames should be multiple of 5!"

    new_visual_emb_frames = []
    max_visual_len = visual_emb_frame.shape[1] * (visual_emb_frame.shape[0] * 0.6)  # keeping 60% frames

    for start_idx in range(0, len(visual_emb_frame), 5):
        end_idx = min(start_idx + window_size, len(visual_emb_frame))
        chunk_feature = visual_emb_frame[start_idx:end_idx]  # 5, HW, C
        if len(chunk_feature) == 1:
            new_visual_emb_frames.append(chunk_feature[0])
            continue

        sim = F.cosine_similarity(
            chunk_feature[0]
            .unsqueeze(0)
            .repeat_interleave(len(chunk_feature[1:]), dim=0),
            chunk_feature[1:],
            dim=-1,
        )
        new_visual_emb_frame = torch.cat(
            [chunk_feature[0],chunk_feature[1:].flatten(0, 1)[sim.flatten(0, 1) < 0.7]],
            dim=0,
        )
        new_visual_emb_frames.append(new_visual_emb_frame)

    reduced_visual_len = sum([x.shape[0] for x in new_visual_emb_frames])
    
    if reduced_visual_len > max_visual_len:
        factor = (reduced_visual_len - max_visual_len) % len(new_visual_emb_frames)
        force_remove = math.ceil(
            (reduced_visual_len - max_visual_len - factor)
            / len(new_visual_emb_frames)
        )

        # force removal 
        for chunk_i in range(len(new_visual_emb_frames)):
            new_visual_emb_frames[chunk_i] = new_visual_emb_frames[chunk_i][:-force_remove]

        # extra removal -- factor
        for _ in range(int(factor)):
            chunk_i = random.randint(0, len(new_visual_emb_frames) - 1)
            new_visual_emb_frames[chunk_i] = new_visual_emb_frames[chunk_i][:-1]
        
        new_visual_emb_frames = torch.cat(new_visual_emb_frames, dim=0)
        new_visual_emb_frames = new_visual_emb_frames[:int(max_visual_len), :]
        
    else:
        # if the video is shorter, keep it intact
        # we would not extract key frames rather just take the 50% alternate frames
        # step = 3
        # new_visual_emb_frames = visual_emb_frame[::step, :, :]  # Slicing to get [50, :, :]
        # new_visual_emb_frames = new_visual_emb_frames.flatten(0,1)

        # 60% frames
        total_frames = visual_emb_frame.shape[0]
        target_frames = int(total_frames * 0.6)
        indices = torch.linspace(0, total_frames - 1, steps=target_frames).round().long()
        new_visual_emb_frames = visual_emb_frame[indices, :].flatten(0, 1)

    new_visual_emb_frames = new_visual_emb_frames.view(
                            int(new_visual_emb_frames.shape[0]/visual_emb_frame.shape[1]),
                            visual_emb_frame.shape[1],
                            new_visual_emb_frames.shape[-1]
                            )
    return new_visual_emb_frames

def get_spatio_temporal_features(features, num_temporal_tokens=20):
    t, s, c = features.shape

    temporal_tokens = torch.mean(features, dim=1).detach().numpy()
    # print("temporal_tokens: ", len(temporal_tokens))
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

    spatial_tokens = torch.mean(features, dim=0).detach().numpy()
    # print("spatial_tokens: ", len(spatial_tokens))
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

    return sp_features

def process_dino_and_vcgpt_files(x, y):
    # Fetch all files from args.dino_path and args.vcgpt_features_path
    args = parse_args()
    dino_path = args.dino_path
    video_path = args.video_path
    dino_files = os.listdir(dino_path)[x:y]
    local_path = args.local_feature_path
    os.makedirs(local_path, exist_ok=True)

    # Initialize the CLIP model
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', 
                                                         torch_dtype=torch.float16)
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', 
                                                   torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).cuda()
    vision_tower.eval()

    # Load pickled tensors
    for file in tqdm.tqdm(dino_files, total=len(dino_files)):
        if not os.path.exists(f"{local_path}/{file}"):
            try:
                dino_tensors = pickle.load(open(f"{dino_path}/{file}", 'rb'))[:,1:,:]
                reduced_tensor = reduce_similar_frames(dino_tensors) # (60, 256, 1024)

                # spatial features from clip forr frames 
                # one for 12 frames
                arr = [19,39,59,79,99]
                frames = load_video(f"{video_path}/{file.replace('.pkl','.mp4')}")
                frames = [frames[i] for i in arr]
                video_tensor = image_processor.preprocess(frames, return_tensors='pt')['pixel_values']
                video_tensor = video_tensor.half().cuda()
                image_forward_outs = vision_tower(video_tensor, output_hidden_states=True)
                select_hidden_state_layer = -2
                select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                batch_features = select_hidden_state[:, 1:] # (5, 256, 1024)
                batch_features = batch_features.repeat(12, 1, 1)[:60]  # (60,:,:)

                # Compute attention scores (dot product between A_expanded and B)
                attn_scores = torch.sum(batch_features * reduced_tensor, dim=-1)  # Shape: [60, 256]

                # Normalize with softmax (across sequence dimension)
                attn_weights = F.softmax(attn_scores, dim=0)  # Shape: [60, 256]

                # Expand attn_weights to match feature dimensions
                attn_weights = attn_weights.unsqueeze(-1)  # Shape: [60, 256, 1]

                # Apply attention to B
                fused_tensor = attn_weights * reduced_tensor  # Shape: [60, 256, 1024]

                # Option 1: Weighted Sum Fusion (Adds CLIP's info into B)
                fused_tensor = fused_tensor + batch_features  # Shape: [60, 256, 1024]

                # Option 2: Concatenation Fusion (Keeps A separate)
                # output = torch.cat([B_fused, A_expanded], dim=-1)  # Shape: [60, 256, 2048]

                # local features
                local_feat = merge_tokens(
                    fused_tensor, 
                    r_merge_list=[2880, 1440, 720, 360, 180, 90, 40]
                ).detach().cpu().numpy().astype("float16")  # [1280, 640, 320, 160, 80, 40, 10]
                with open(f"{local_path}/{file}", 'wb') as f:
                    pickle.dump(local_feat, f)
        
            except Exception as e:
                print(f"Can't process {video_path} due to {e}")

if __name__ == "__main__":
    args = parse_args()
    x, y = args.xy.split("-")
    process_dino_and_vcgpt_files(int(x), int(y))

    

# git pull; python scripts/save_features.py --video_path /data/shared/gauravs/llapsa/vcgpt_clips/ --dino_path /data/shared/gauravs/llapsa/llapsa_encoded_video_clips/dino_features/dino_features/ --local_feature_path /data/shared/gauravs/llapsa/llapsa_encoded_video_clips/local_features --global_feature_path /data/shared/gauravs/llapsa/llapsa_encoded_video_clips/global_features --xy 0-10