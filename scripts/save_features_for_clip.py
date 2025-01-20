import os, json
import pickle
import argparse
import torch
import tqdm
from torch.nn import MultiheadAttention
from transformers import BertTokenizer, BertModel, BertConfig

def parse_args():
    parser = argparse.ArgumentParser(description="getting text fragments")
    parser.add_argument("--dino_path", required=True, help="Path to DinoV2 tensors")
    parser.add_argument("--text_path", required=True, help="Path to 6sec text fragments")
    parser.add_argument("--text_option", default=0, help="How to treat null patches")
    parser.add_argument("--save_local_features_dir", required=True, help="directory to save the local features")
    parser.add_argument("--save_global_features_dir", required=True, help="directory to save the global features")
    args = parser.parse_args()
    return args

def split_tensor(tnsr):
    num_sub_tensors = 10
    
    # Ensure we have enough elements to create 10 sub-tensors
    if tnsr.shape[0] % num_sub_tensors != 0:
        raise ValueError(f"DINO tensor length {tnsr.shape[0]} is not divisible by {num_sub_tensors}")
    
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
    custom_config.hidden_size = 1536
    custom_config.intermediate_size = custom_config.hidden_size * 4  # Standard in BERT
    custom_config.num_attention_heads = 12  # Ensure divisibility of hidden_size by num_attention_heads

    # Initialize a model with the custom configuration
    model = BertModel(custom_config)
    model.eval()

    dino_files = [f for f in os.listdir(args.dino_path) 
                  if f.replace(".pkl", ".json") in os.listdir(args.text_path)]

    for fyl in tqdm.tqdm(dino_files, total=len(dino_files)):
        # process dino tensor
        dpath = f"{args.dino_path}/{fyl}"
        output_path = f"{args.save_dir}/{fyl.replace('.mp4','.pkl')}"

        if not os.path.exists(output_path):
            with open(dpath, 'rb') as _f:
                dino_tensor = pickle.load(_f).to(torch.float32)  # (100,:,:)
            
            split_dino_tensors = split_tensor(dino_tensor) # a list of [10,:,:] * N=10

            # process text fragments using Bert
            with open(f"{args.text_path}/{fyl.replace('.pkl','.json')}", "rb") as _f:
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
            
            final_tnsr = torch.cat(cross_attn_outputs, dim=0)
            with open(output_path, 'wb') as f:
                pickle.dump(final_tnsr, f)

if __name__ == "__main__":
    main()