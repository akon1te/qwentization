import argparse
import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def dequantize_awq_layer(awq_layer):
    in_features = awq_layer.in_features
    identity = torch.eye(in_features, dtype=torch.float16, device=awq_layer.device)
    
    weights = []
    batch_size = 128 
    
    with torch.no_grad():
        for i in range(0, in_features, batch_size):
            batch = identity[i : i + batch_size]
            out = awq_layer(batch)
            weights.append(out)
            
    W_fp16 = torch.cat(weights, dim=0)
    return W_fp16.t()


def apply_svd_to_head(model, rank_ratio=0.8):
    lm_head = model.lm_head
    
    if not isinstance(lm_head, nn.Linear):
        W = dequantize_awq_layer(lm_head)
    else:
        W = lm_head.weight.data

    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    
    full_rank = len(S)
    target_rank = int(full_rank * rank_ratio)
    
    U_r = U[:, :target_rank]
    S_r = S[:target_rank]
    Vh_r = Vh[:target_rank, :]
    
    sqrt_S = torch.diag(torch.sqrt(S_r))
    
    W_down = torch.matmul(sqrt_S, Vh_r).to(W.device).half()
    W_up = torch.matmul(U_r, sqrt_S).to(W.device).half()
    
    linear_down = nn.Linear(W.shape[1], target_rank, bias=False)
    linear_up = nn.Linear(target_rank, W.shape[0], bias=False)
    
    linear_down.weight.data = W_down
    linear_up.weight.data = W_up
    
    new_head = nn.Sequential(linear_down, linear_up)
    model.lm_head = new_head
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--rank_ratio", type=float, default=0.8)
    
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = apply_svd_to_head(model, rank_ratio=args.rank_ratio)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    try:
        model.save_pretrained(args.output_path)
    except Exception as e:
        print(f"Warning: save_pretrained failed: {e}")

    tokenizer.save_pretrained(args.output_path)
    
    pt_path = os.path.join(args.output_path, "model.pt")
    torch.save(model, pt_path)

if __name__ == "__main__":
    main()
