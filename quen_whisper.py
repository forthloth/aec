import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

# 假设引用路径保持不变，这里仅保留你的导入结构
from .quen import QwenEncoder as quen
from .whisper_base import Encoder as whisper
from .configuration_qwen3 import Qwen3Config
from .modeling_qwen3 import Qwen3Attention

def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype):
    """
    将 [B, L] 的 0/1 (或 bool) mask 转换为 [B, 1, 1, L] 的加性 mask
    1 (True)  -> 0.0
    0 (False) -> -inf
    """
    # 1. 确保是布尔类型，方便后续取反操作
    mask_bool = mask.to(torch.bool)
    
    # 2. 扩展维度到 [B, 1, 1, L]
    # 注意：在 Encoder 中，我们通常只需要广播最后的维度
    expanded_mask = mask_bool[:, None, None, :]
    
    # 3. 创建一个全 0 的张量，并在无效位置（~expanded_mask）填充负无穷
    # 这样：True 的地方保持 0.0，False 的地方变成 -inf
    return torch.zeros(expanded_mask.shape, device=mask.device, dtype=dtype).masked_fill(
        ~expanded_mask, torch.finfo(dtype).min
    )
class quen_whisper_Encoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.whisper_encoder = quen(**kwargs)
        self.quen_encoder = whisper(**kwargs)
        assert self.whisper_encoder.output_dim == self.quen_encoder.output_dim, "whisper and quen output dim must be equal"
        
        self.output_dim = self.whisper_encoder.output_dim
        self.norm1 = nn.LayerNorm(self.output_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.output_dim, elementwise_affine=False, eps=1e-6)
        
        self.whisper_modulation = nn.Linear(self.output_dim, 2 * self.output_dim, bias=True)
        self.quen_modulation = nn.Linear(self.output_dim, 2 * self.output_dim, bias=True)

        # === 1. 初始化配置与必要的参数 ===
        config = Qwen3Config()

        custom_settings = {
            "hidden_size": self.output_dim,
            "num_attention_heads": 8,
            "num_key_value_heads": 8, 
            "intermediate_size": self.output_dim,
            
            # 门控机制设定
            "elementwise_attn_output_gate": True,
            "headwise_attn_output_gate": False,  # 源码中判断了此项，建议显式声明
            
            # 修复报错的关键：偏置项
            "qkv_bias": True,                    # 这里设为 True 或 False 取决于你的模型设计
            
            # 位置编码与 Dropout
            "max_position_embeddings": 2250, 
            "rope_theta": 10000.0, 
            "attention_dropout": 0.0,

            # 归一化相关（源码中 if self.use_qk_norm用到了）
            "use_qk_norm": False,                # 如果设为 True，则必须补充 rms_norm_eps
            "rms_norm_eps": 1e-6,
        }
        config.update(custom_settings)
        self.fusion_attn = Qwen3Attention(config, layer_idx=0)

        # === 2. 新增：模态区分的可学习 Token ===
        # 形状 [1, 1, C]，利用广播机制加到整个序列上
        self.whisper_modality_token = nn.Parameter(torch.zeros(1, 1, self.output_dim))
        self.quen_modality_token = nn.Parameter(torch.zeros(1, 1, self.output_dim))
        
        # 初始化这些 token (可选，用正态分布初始化更佳)
        nn.init.normal_(self.whisper_modality_token, std=0.02)
        nn.init.normal_(self.quen_modality_token, std=0.02)
        nn.init.constant_( self.whisper_modality_token, 0.0)
        nn.init.constant_( self.quen_modality_token, 0.0)
        # state_dict = torch.load('/asdata/lmd/aec/whisper_quen_gate_transformer_step40000.pth', weights_only=True)
        # #set device to be consistent with the model's parameters
        # # device = next(self.parameters()).device
        # # C. 将参数注入模型
        # self.load_state_dict(state_dict)

    def forward(self, input_features, attention_mask=None):
        with torch.no_grad():
            whisper_outputs, whisper_mask = self.whisper_encoder(input_features, attention_mask)
            quen_outputs, quen_mask = self.quen_encoder(input_features, attention_mask)
            
            whisper_mask = whisper_mask.bool().to(whisper_outputs.device)
            quen_mask = quen_mask.bool().to(quen_outputs.device)

        # --- Whisper 处理 ---
        B, N, C = whisper_outputs.shape
        whisper_outputs = whisper_outputs.reshape(B * N, C)
        whisper_scale, whisper_shift = self.whisper_modulation(whisper_outputs).chunk(2, dim=-1)
        whisper_outputs = (whisper_scale+1) * self.norm1(whisper_outputs) + whisper_shift
        whisper_outputs = whisper_outputs.reshape(B, N, C)
        
        # --- Quen 处理 ---
        B, N2, C2 = quen_outputs.shape
        quen_outputs = quen_outputs.reshape(B * N2, C2)
        quen_scale, quen_shift = self.quen_modulation(quen_outputs).chunk(2, dim=-1)
        quen_outputs = (quen_scale+1) * self.norm2(quen_outputs) + quen_shift
        quen_outputs = quen_outputs.reshape(B, N2, C2)

        # === 3. 新增：添加模态 Token (Position Information Supplement) ===
        # 在拼接前，给不同来源加上唯一的可学习标识，代替简单的绝对位置编码
        whisper_outputs = whisper_outputs + self.whisper_modality_token
        quen_outputs = quen_outputs + self.quen_modality_token

        # === 4. 拼接输出与 Mask ===
        # 在 dim=1 (Time) 维度拼接
        combined_outputs = torch.cat((whisper_outputs, quen_outputs), dim=1)
        combined_mask = torch.cat((whisper_mask, quen_mask), dim=1)

        # === 5. 生成 Position IDs (基于时间对齐逻辑) ===
        # Whisper: [0, 1, ..., N-1], Quen: [0, 1, ..., N2-1]
        # 因为它们时间步相同，索引也应该相同
        p_ids_w = torch.arange(N, device=combined_outputs.device)
        p_ids_q = torch.arange(N2, device=combined_outputs.device)
        # 拼接后的索引序列，例如 [0,1,2,0,1,2] 表示两段序列在时间步上是重合的
        combined_p_ids = torch.cat((p_ids_w, p_ids_q), dim=0).unsqueeze(0).expand(B, -1)

        # === 5. 调用 Rotary Embedding ===
        # 根据你提供的源码：forward(self, x, position_ids)
        cos, sin = self.fusion_attn.rotary_emb(combined_outputs, combined_p_ids)
        position_embeddings = (cos, sin)


        # === 6. Mask 格式转换 ===
        # 将 [B, Total_Len] (0/1) -> [B, 1, 1, Total_Len] (0.0/-inf)
        attention_mask_4d = _prepare_4d_attention_mask(combined_mask, combined_outputs.dtype)

        # === 7. Attention 计算 ===
        final_result, _, _ = self.fusion_attn(
            hidden_states=combined_outputs,
            attention_mask=attention_mask_4d,
            position_embeddings=position_embeddings,
            output_attentions=False
        )

        return final_result, combined_mask

if __name__ == "__main__":
    import sys
    import os
    # 路径配置
    sys.path.append("/home/bms34/data/lmd/xares-llm-main")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    from example.Quen2_whisper2.quen import QwenEncoder as quen
    from example.Quen2_whisper2.whisper_base import Encoder as whisper
    from example.Quen2_whisper2.configuration_qwen3 import Qwen3Config
    from example.Quen2_whisper2.modeling_qwen3 import Qwen3Attention
    from example.Quen2_whisper.quen import QwenEncoder
    from example.Quen2_whisper.whisper_base import Encoder
    from src.xares_llm.audio_encoder_checker import check_audio_encoder

    enc = quen_whisper_Encoder()
    # 检查
    check_audio_encoder(enc)