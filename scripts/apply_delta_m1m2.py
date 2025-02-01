"""
Usage:
python TODO: Add usage
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from longvlm.model.longvlm import LongVLMForCausalLM


def apply_delta(base_model_path, target_model_path, delta_path):
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Loading delta")
    delta = LongVLMForCausalLM.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path)

    print("Applying delta")
    base_sd = base.state_dict()
    delta_sd = delta.state_dict()

    for name, param in tqdm(delta_sd.items(), desc="Applying delta"):
        if name not in base_sd:
            # Replace mm_projector with mm1 and mm2
            if name == 'model.mm1.weight':
                print("Replacing mm_projector.weight → mm1.weight")
                param.data.copy_(base_sd['model.mm_projector.weight'])

            elif name == 'model.mm1.bias':
                print("Replacing mm_projector.bias → mm1.bias")
                param.data.copy_(base_sd['model.mm_projector.bias'])

            elif name == 'model.mm2.weight' or name == 'model.mm2.bias':
                print(f"{name} is a new layer and will be randomly initialized.")

            else:
                assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f'{name} not in base model'
            continue
        
        # Apply delta normally
        if param.data.shape == base_sd[name].shape:
            param.data += base_sd[name]
        else:
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], \
                f'{name} dimension mismatch: {param.data.shape} vs {base_sd[name].shape}'
            bparam = base_sd[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] += bparam

    print("Saving target model")
    delta.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)



