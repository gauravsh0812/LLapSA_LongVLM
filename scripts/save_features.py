import os 
import numpy as np
import math
import random
import torch
import tqdm
import pickle
import argparse
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from transformers import SiglipImageProcessor, SiglipVisionModel
from scripts.base_encoder import BaseVisionTower
from longvlm.model.merge import merge_tokens

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--xy", required=True)
    parser.add_argument("--dino_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    return args

class SiglipVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name="google/siglip-so400m-patch14-384", delay_load=False):
        super(SiglipVisionTower, self).__init__(vision_tower_name, delay_load)
        
        model_path = vision_tower_name
        base_model_name, res, interp = model_path, 384, 576
        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 512
        self._interp_size = interp
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self._hidden_size = 1024

    def load_model(self, device_map=None):
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.config.hidden_size
        self._image_size = self.vision_tower.config.image_size
        self._patch_size = self.vision_tower.config.patch_size
        self.image_processor = SiglipImageProcessor.from_pretrained(
            self.vision_tower_name
        )

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def forward(self, images,):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True, interpolate_pos_encoding=True,
            ).hidden_states[-1]
            return image_features

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
    dino_files = os.listdir(dino_path)[x:y]
    output_path = args.output_path

    print("dino path: ", dino_path)

    siglip = SiglipVisionTower()

    # Load pickled tensors
    for file in tqdm.tqdm(dino_files, total=len(dino_files)):
        # if not os.path.exists(f"{output_path}/{file}"):
        #     try:
        dino_tensors = pickle.load(open(f"{dino_path}/{file}", 'rb'))
        print("dino: ", dino_tensors.shape)
        reduced_tensor = reduce_similar_frames(dino_tensors) # (15360, 1024)
        print("reduced: ", reduced_tensor.shape)
        siglip_tensor = siglip(reduced_tensor)
        print(siglip_tensor)
        
        
        # exit()
        # spatial_tokens = get_spatio_temporal_features(reduced_tensor)
        # print(spatial_tokens.shape)
        # siglip_tensor = siglip(reduced_tensor)
        # siglip_tensor_np = siglip_tensor.numpy()
        # spatial_tokens = torch.mean(reduced_tensor, dim=0).detach().numpy()
        # print(spatial_tokens.shape)
        # with open(f"{output_path}/{file}", 'wb') as f:
        #     pickle.dump(spatial_tokens, f)
            # except:
            #     print(f"{file} doesn't work!")
        
if __name__ == "__main__":
    args = parse_args()
    x, y = args.xy.split("-")
    process_dino_and_vcgpt_files(int(x), int(y))


# Example usage:
# embeddings = np.random.rand(100, 512)  # Example CLIP embeddings
# unique_indices = calculate_average_similarity(embeddings)
# filtered_embeddings = embeddings[unique_indices]