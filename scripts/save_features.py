import os 
import numpy as np
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
    def __init__(self, vision_tower_name, delay_load=False):
        super(SiglipVisionTower, self).__init__(vision_tower_name, delay_load)
        
        model_path = "google/siglip-so400m-patch14-384"
        base_model_name, res, interp = model_path, 384, 576
        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 512
        self._interp_size = interp
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self._hidden_size = 1152

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


def reduce_similar_frames(embeddings, window_size=5, similarity_threshold=0.8):
    """
    Eliminate frames with high average similarity within a sliding window.
    
    Args:
        embeddings (numpy.ndarray): A 2D array where each row is an embedding vector.
        window_size (int): Number of frames in each window (J in the formula).
        similarity_threshold (float): Threshold for average similarity to reduce frames.
        
    Returns:
        list: Indices of frames to keep.
    """
    num_frames = len(embeddings)
    to_keep = []
    
    for start in range(0, num_frames, window_size):
        # Select window
        end = min(start + window_size, num_frames)
        window_embeddings = embeddings[start:end]
        
        # Compute cosine similarity matrix for the window
        print(window_embeddings.shape)
        window_embeddings = window_embeddings.view(window_embeddings.shape[0], -1)
        window_embeddings = window_embeddings.cpu().numpy()
        similarity_matrix = cosine_similarity(window_embeddings)
        np.fill_diagonal(similarity_matrix, 0) # Ignore self-similarity
        
        # Calculate average similarity for each image
        avg_similarity = np.mean(similarity_matrix, axis=1)
        print(avg_similarity)
        
        # Select images with average similarity below the threshold
        to_keep_indices = np.where(avg_similarity < similarity_threshold)[0]
        to_keep.append(to_keep_indices)
    
    print("to keep: ", to_keep)
    
    # return embeddings[to_keep]

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

    # Load pickled tensors
    for file in tqdm.tqdm(dino_files, total=len(dino_files)):
        # if not os.path.exists(f"{output_path}/{file}"):
        #     try:
        dino_tensors = pickle.load(open(f"{dino_path}/{file}", 'rb'))[:, 1:]
        reduced_tensor = reduce_similar_frames(dino_tensors) # (20, 256, 1024)
        print(reduced_tensor)
        exit()
        spatial_tokens = get_spatio_temporal_features(reduced_tensor)
        # print(spatial_tokens.shape)
        # siglip_tensor = siglip(reduced_tensor)
        # siglip_tensor_np = siglip_tensor.numpy()
        # spatial_tokens = torch.mean(reduced_tensor, dim=0).detach().numpy()
        # print(spatial_tokens.shape)
        with open(f"{output_path}/{file}", 'wb') as f:
            pickle.dump(spatial_tokens, f)
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