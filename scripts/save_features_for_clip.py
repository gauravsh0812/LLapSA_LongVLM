import os, json
import pickle
import argparse
import torch
import tqdm
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torch.nn import MultiheadAttention
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import CLIPVisionModel, CLIPImageProcessor
from longvlm.model.merge import merge_tokens

def parse_args():
    parser = argparse.ArgumentParser(description="getting text fragments")
    parser.add_argument("--video_path", required=True, help="Path to videos")
    parser.add_argument("--text_path", required=True, help="Path to 6sec text fragments")
    parser.add_argument("--text_option", default=0, help="How to treat null patches")
    parser.add_argument("--save_local_features_dir", required=True, help="directory to save the local features")
    parser.add_argument("--save_global_features_dir", required=True, help="directory to save the global features")
    
    args = parser.parse_args()
    return args

def load_video(vis_path, num_frm=100):
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    # a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    
    if img_array.shape[0] != num_frm:
        img_array = torch.from_numpy(img_array).permute(1, 2, 3, 0).float()
        img_array = torch.nn.functional.interpolate(img_array, size=num_frm)
        img_array = img_array.permute(3, 0, 1, 2).to(torch.uint8).numpy()
    
    img_array = img_array.reshape((1, num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(num_frm):
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

def split_tensor(tnsr):
    num_sub_tensors = 10
    print(tnsr.shape)
    # Ensure we have enough elements to create 10 sub-tensors
    if tnsr.shape[0] % num_sub_tensors != 0:
        raise ValueError(f"image tensor length {tnsr.shape[0]} is not divisible by {num_sub_tensors}")
    
    sub_tensor_size = tnsr.shape[0] // num_sub_tensors

    sub_tensors = []
    for i in range(num_sub_tensors):
        start_idx = i * sub_tensor_size
        end_idx = (i + 1) * sub_tensor_size
        sub_tensor = tnsr[start_idx:end_idx]
        sub_tensors.append(sub_tensor)
    return sub_tensors

def cross_attention(image_tensor, text_tensor,):
    
    # Define dimensions
    hidden_dim = image_tensor.shape[-1]
    num_heads = 6

    text_tensor = text_tensor.repeat_interleave(repeats=10, dim=0)  # Shape [10, 1, 1536]

    # MultiheadAttention module
    mha = MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True).cuda()
    for param in mha.parameters():
        param.requires_grad = False

    # Compute cross-attention
    # Queries are from the image, keys/values are from the text
    text_tensor = text_tensor.cuda()
    image_tensor = image_tensor.cuda()
    output, attention_weights = mha(query=image_tensor, key=text_tensor, value=text_tensor)
    return output

def main():
    args = parse_args()

    os.makedirs(args.save_local_features_dir, exist_ok=True)
    os.makedirs(args.save_global_features_dir, exist_ok=True)
    
    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create a new BERT configuration with a larger hidden size
    custom_config = BertConfig.from_pretrained('bert-base-uncased')
    custom_config.hidden_size = 1024
    custom_config.num_attention_heads = 8
    custom_config.num_hidden_layers = 8
    custom_config.intermediate_size = custom_config.hidden_size * 4  

    # Initialize a model with the custom configuration
    model = BertModel(custom_config)
    model.eval()

    for n, p in model.named_parameters():
        p.requires_grad_(False)

    # vision model clip
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', 
                                                         torch_dtype=torch.float16)
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', 
                                                   torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True)

    vision_tower.cuda()
    vision_tower.eval()
    for n, p in vision_tower.named_parameters():
        p.requires_grad_(False)


    video_files = [f for f in os.listdir(args.video_path) 
                  if f.replace(".mp4", ".json") in os.listdir(args.text_path)][:10]
    
    for fyl in tqdm.tqdm(video_files, total=len(video_files)):
        video_id = fyl.split('.')[0]
        local_feat_path = f"{args.save_local_features_dir}/{video_id}.pkl"
        global_feat_path = f"{args.save_global_features_dir}/{video_id}.pkl"

        if (not os.path.exists(local_feat_path)) or (not os.path.exists(global_feat_path)) :
            # try:
                video_path = f"{args.video_path}/{fyl}"
                video = load_video(video_path)
                video_tensor = image_processor.preprocess(video, return_tensors='pt')['pixel_values']
                video_tensor = video_tensor.half().cuda()
                with torch.no_grad():
                    image_forward_outs = vision_tower(video_tensor, output_hidden_states=True).hidden_states[-2][:, 1:]

                split_dino_tensors = split_tensor(image_forward_outs) # a list of [10,:,:] * N=10

                # process text fragments using Bert
                with open(f"{args.text_path}/{video_id}.json", "rb") as _f:
                    text_json = json.load(_f)

                cross_attn_outputs = []
                # print(text_json)
                # print(text_json.keys())
                for i, k in enumerate(["0-6","6-12","12-18","18-24","24-30",
                        "30-36","36-42","42-48","48-54","54-60"]):
                    # print(k)
                    if k in text_json.keys():
                        text = text_json[k]
                        # print(text.split())
                        if text == "null" or text == None or text == "":
                            if args.text_option == 1:
                                last_hidden_state = torch.ones(1,1, 1536)
                            elif args.text_option == 0:
                                last_hidden_state = torch.zeros(1,1, 1536)
                        else:
                            inputs = tokenizer(text, return_tensors='pt', max_length=512, padding=True, truncation=True)
                            with torch.no_grad():
                                outputs = model(**inputs)
                            last_hidden_state = outputs.last_hidden_state

                    else:
                        if args.text_option == 1:
                            last_hidden_state = torch.ones(1,1, 1536)
                        elif args.text_option == 0:
                            last_hidden_state = torch.zeros(1,1, 1536)


                    # cross attention
                    output = cross_attention(split_dino_tensors[i], last_hidden_state)
                    cross_attn_outputs.append(output)
                
                final_ca_output = torch.cat(cross_attn_outputs, dim=0)

                # merging

                # local features
                local_feat = merge_tokens(
                    final_ca_output.hidden_states[-2][:, 1:], 
                    r_merge_list=[2880, 1440, 720, 360, 180, 90, 40]
                ).detach().cpu().numpy().astype("float16")  # [1280, 640, 320, 160, 80, 40, 10]
                    
                with open(local_feat_path, 'wb') as f:
                    pickle.dump(local_feat, f)

                # global_features 
                global_feat = torch.cat(
                    [mem[:, :1] for mem in final_ca_output.hidden_states], 
                    dim=1).mean(0).squeeze(0).detach().cpu().numpy().astype("float16")
                with open(global_feat_path, 'wb') as f:
                    pickle.dump(global_feat, f)

            # except Exception as e:
            #     print(f"Can't process {video_path}: {e}")


if __name__ == "__main__":
    main()