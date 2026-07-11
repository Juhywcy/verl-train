import os
# 设置环境变量，指定使用 4-7 号 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import torch
import torch.nn.functional as F

class CustomLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, clip_value):
        # logits: (B, V)
        # labels: (B,)
        
        # 标准前向传播: log_softmax 和 gather
        log_probs = F.log_softmax(logits, dim=-1)
        ctx.save_for_backward(log_probs, labels)
        ctx.clip_value = clip_value
        
        selected_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
        return selected_log_probs

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output 是 dL/d(log_prob_selected)
        # 我们想要 dL/d(logits)
        
        log_probs, labels = ctx.saved_tensors
        clip_value = ctx.clip_value
        
        probs = torch.exp(log_probs)
        grad_output_expanded = grad_output.unsqueeze(-1) # (B, 1)
        
        # 1. 计算基础梯度项: - p_j * g
        # 这是 softmax 部分对所有索引的梯度贡献
        grad_logits = - probs * grad_output_expanded
        
        if clip_value is not None:
             print(f"   [Backward] 应用截断值: {clip_value}")
             # 2. 截断所有梯度
             grad_logits_clamped = torch.clamp(grad_logits, min=-clip_value, max=clip_value)
             
             # 3. 将目标索引恢复为原始（未截断）值
             # 根据此逻辑，我们不希望截断正确标签的梯度
             target_indices = labels.unsqueeze(-1)
             original_target_grads = grad_logits.gather(-1, target_indices)
             grad_logits_clamped.scatter_(-1, target_indices, original_target_grads)
             
             grad_logits = grad_logits_clamped

        # 4. 将 g 加到目标索引
        # log_softmax(x)_y 对 x_y 的梯度是 (1 - p_y)
        # 我们目前在 grad_logits[y] 中有 -p_y * g
        # 我们需要加上 g 使其变为 g * (1 - p_y)
        grad_logits.scatter_add_(-1, labels.unsqueeze(-1), grad_output_expanded)
        
        return grad_logits, None, None

def test_gradient_control():
    # 确保结果可复现
    torch.manual_seed(42)
    
    # 检查是否有可用的 GPU，如果有则将数据移动到 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")

    B, V = 2, 5
    # 注意：在创建叶子节点 tensor 时，如果需要梯度，最好直接在创建时指定 device 和 requires_grad
    logits = torch.randn(B, V, device=device, requires_grad=True)
    labels = torch.randint(0, V, (B,), device=device)
    
    print(f"Logits 形状: {logits.shape}")
    print(f"Labels: {labels}")

    print("\n--- 1. 标准 PyTorch Autograd ---")
    # clone().detach() 会保留数据但切断历史，requires_grad_(True) 开启新梯度记录
    logits_std = logits.clone().detach().requires_grad_(True)
    log_probs_std = F.log_softmax(logits_std, dim=-1)
    selected_log_probs_std = torch.gather(log_probs_std, -1, labels.unsqueeze(-1)).squeeze(-1)
    loss_std = selected_log_probs_std.sum()
    loss_std.backward()
    print("梯度 (标准):\n", logits_std.grad)
    
    print("\n--- 2. 自定义 Autograd 带干预 (无截断) ---")
    logits_custom = logits.clone().detach().requires_grad_(True)
    # clip_value = None 意味着它的行为应该与标准完全一致
    selected_log_probs_custom = CustomLogProb.apply(logits_custom, labels, None)
    loss_custom = selected_log_probs_custom.sum()
    loss_custom.backward()
    print("梯度 (自定义 无截断):\n", logits_custom.grad)
    
    match = torch.allclose(logits_std.grad, logits_custom.grad)
    print(f"匹配 (标准 vs 自定义 无截断): {match}")

    print("\n--- 3. 自定义 Autograd 带干预 (有截断) ---")
    logits_clipped = logits.clone().detach().requires_grad_(True)
    clip_val = 0.05 # 小的截断值以确保发生截断
    selected_log_probs_clipped = CustomLogProb.apply(logits_clipped, labels, clip_val)
    loss_clipped = selected_log_probs_clipped.sum()
    loss_clipped.backward()
    print(f"梯度 (自定义 截断 {clip_val}):\n", logits_clipped.grad)
    
    diff = not torch.allclose(logits_std.grad, logits_clipped.grad)
    print(f"存在差异 (标准 vs 自定义 截断): {diff}")
    
    # 手动验证截断逻辑
    probs = F.softmax(logits, dim=-1) # 使用原始 logits 计算概率用于验证
    print("\n--- 验证 ---")
    print(f"非目标 token 的预期截断梯度应在范围 [{-clip_val}, {clip_val}] 内")
    
    # 检查非目标梯度是否被截断
    grad = logits_clipped.grad
    mask = torch.ones_like(grad, dtype=torch.bool)
    mask.scatter_(-1, labels.unsqueeze(-1), False)
    non_target_grads = grad[mask]
    
    print("非目标梯度:\n", non_target_grads)
    is_clamped = (non_target_grads.abs() <= clip_val + 1e-5).all()
    print(f"非目标梯度是否被截断到 <= {clip_val}? {is_clamped.item()}")
    
    # 检查目标梯度是否未被截断（或者至少与截断逻辑不同，如果它们很大的话）
    # 在此逻辑中，目标梯度被恢复为原始 -p*g 然后加上 g。
    # 所以目标梯度应该是 g * (1 - p)。
    # 让我们检查目标梯度是否匹配标准梯度（因为我们恢复了它）
    
    target_grads_clipped = grad.gather(-1, labels.unsqueeze(-1))
    target_grads_std = logits_std.grad.gather(-1, labels.unsqueeze(-1))
    
    print("目标梯度 (截断运行):\n", target_grads_clipped.squeeze())
    print("目标梯度 (标准运行):\n", target_grads_std.squeeze())
    
    targets_match = torch.allclose(target_grads_clipped, target_grads_std)
    print(f"目标梯度是否匹配标准? {targets_match}")
    print("(它们应该匹配，因为代码在添加 g 之前显式恢复了目标梯度)")

if __name__ == "__main__":
    test_gradient_control()
