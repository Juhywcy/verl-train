import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_step_entropy_percentile(model, tokenizer, prompt, max_new_tokens=1000, temperature=0.7):
    """
    输入: 
        model, tokenizer: 已加载的模型
        prompt: 输入问题 (str)
    返回:
        每个 reasoning step 首 token 的熵百分位平均值 (float)
    """

    # ===== 1. Encode & Generate =====
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=True
        )

    # ===== 2. 计算每个生成 token 的熵 =====
    all_scores = torch.stack(outputs.scores, dim=1)  # [1, seq_len, vocab_size]
    probs = torch.nn.functional.softmax(all_scores, dim=-1)
    eps = 1e-9
    entropy = (-probs * (probs + eps).log()).sum(dim=-1).squeeze(0).cpu().numpy()

    # ===== 3. 解码文本 =====
    generated_tokens = outputs.sequences[0][inputs.input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_tokens)

    # ===== 4. 以 "\n\n" 分割 reasoning steps =====
    steps = [s for s in generated_text.split("\n\n") if s.strip()]

    # ===== 5. 找每个 step 的首 token =====
    decoded_tokens = tokenizer.convert_ids_to_tokens(generated_tokens)
    entropies = entropy
    percentiles = [np.sum(entropies <= e) / len(entropies) * 100 for e in entropies]

    step_percentiles = []
    for step in steps:
        step_enc = tokenizer(step, add_special_tokens=False)
        if len(step_enc.input_ids) == 0:
            continue
        step_tok = step_enc.input_ids[0]
        try:
            idx_in_gen = generated_tokens.tolist().index(step_tok)
            step_percentiles.append(percentiles[idx_in_gen])
        except ValueError:
            continue

    if len(step_percentiles) == 0:
        return None

    return float(np.mean(step_percentiles))


# ========== 示例：批量计算多个问题 ==========
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    model_name = "/home/hrzhu/model/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    prompts = [
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "If a train travels 60 miles per hour, how long does it take to go 180 miles?",
    ]

    results = []
    for p in prompts:
        avg_percentile = compute_step_entropy_percentile(model, tokenizer, p)
        print(f"\nPrompt: {p}")
        print(f"Average step entropy percentile: {avg_percentile:.2f}%")
        results.append(avg_percentile)
