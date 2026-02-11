# from example.Quen2_whisper.quen import QwenEncoder
# from example.Quen2_whisper.whisper_base import Encoder
import torch
import math
from .quen import QwenEncoder as quen
from .whisper_base import Encoder as whisper

import torch
import torch.nn.functional as F

def get_vad_mask(
    wav: torch.Tensor, 
    sample_rate: int = 16000, 
    frame_ms: float = 10.0, 
    threshold_db: float = -45.0, 
    min_speech_ms: float = 50.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    基于能量的语音活动检测 (VAD) Mask 生成函数。
    
    Args:
        wav (torch.Tensor): 输入音频，形状为 (B, L)。
        sample_rate (int): 采样率，默认 16000。
        frame_ms (float): 计算能量的帧长 (毫秒)，默认 10ms。
        threshold_db (float): 相对静音阈值 (dB)，低于 (最大音量 + 此阈值) 的被视为静音。
        min_speech_ms (float): 最小语音保持时间。用于平滑 Mask，填补语音中的微小空隙 (类似 VAD 的 hangover)。
                               设为 0 则不进行平滑 (与原代码行为一致)，建议设为 50ms-100ms。
        eps (float): 防止 log(0) 的极小值。

    Returns:
        torch.Tensor: Boolean Mask，形状为 (B, L)。True 表示语音，False 表示静音。
    """
    
    B, L = wav.shape
    device = wav.device
    
    # 1. 计算帧长 (Samples)
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len < 1:
        #generate a mask of all True if frame_len is less than 1 sample (e.g., frame_ms < 0.0625ms at 16kHz)
        return torch.ones_like(wav, dtype=torch.bool)

    # 2. Padding (补齐末尾)
    # 为了方便 reshape 或 unfold，我们需要将长度补齐为 frame_len 的整数倍
    # 这样可以避免 unfold 丢弃末尾不足一帧的数据
    remainder = L % frame_len
    pad_len = 0
    if remainder > 0:
        pad_len = frame_len - remainder
        # 使用 Constant Padding 补 0
        wav_padded = F.pad(wav, (0, pad_len), mode='constant', value=0)
    else:
        wav_padded = wav

    # 3. 变形为 (B, N_frames, Frame_len)
    # 这里使用 reshape 代替 unfold，因为我们的步长(hop)等于帧长(frame)，即无重叠
    # 相比 unfold，reshape 在这种情况下更快且内存占用更小
    # 新形状: [B, num_frames, frame_len]
    wav_frames = wav_padded.view(B, -1, frame_len)
    
    # 4. 计算能量 (Energy / RMS)
    # 原公式: 20 * log10(sqrt(mean(x^2)))
    # 等价于: 10 * log10(mean(x^2)) -> 省略开根号，计算更少
    energy = wav_frames.pow(2).mean(dim=-1) # [B, num_frames]
    
    # 转换为 dB
    db_frames = 10.0 * torch.log10(energy + eps)
    
    # 5. 动态阈值判定 (与原代码逻辑一致：基于该样本最大音量的相对阈值)
    # 获取每个样本的最大 dB 值: [B, 1]
    max_db = db_frames.amax(dim=1, keepdim=True)
    # 计算相对 dB: [B, num_frames]
    rel_db = db_frames - max_db
    
    # 生成粗略 Mask: [B, num_frames]
    mask_frames = rel_db >= threshold_db

    # 6. 平滑处理 (Smoothing / Hangover) - 这是一个"更合理"的改进
    # 原代码是硬切分，容易把单词中间的弱音切掉。
    # 这里使用 1D MaxPool 模拟"膨胀"操作，只要窗口内有一帧是 True，周围就变成 True。
    if min_speech_ms > 0:
        # 计算需要覆盖多少帧
        kernel_size = int(min_speech_ms / frame_ms)
        if kernel_size > 1:
            # MaxPool 需要 float 输入，且形状为 [B, C, L]
            mask_float = mask_frames.float().unsqueeze(1) 
            # Padding 保证卷积后长度不变 (Same padding)
            padding = kernel_size // 2
            
            # 使用 MaxPool1d 进行膨胀 (Dilation)
            # 这会填补小于 min_speech_ms 的静音缝隙，并稍微向两端扩展语音边界
            mask_dilated = F.max_pool1d(
                mask_float, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=padding
            )
            
            # 裁剪掉多余的 padding (因为 max_pool 的 padding 处理可能导致尺寸微变)
            mask_dilated = mask_dilated[:, :, :mask_frames.shape[1]]
            
            # 转回 Bool
            mask_frames = mask_dilated.squeeze(1) > 0.5

    # 7. 上采样还原回 Sample 级别 (Upsampling)
    # 将 Frame 级别的 Mask [B, N] 扩展回 [B, N * frame_len]
    # repeat_interleave 是处理这种非重叠窗口还原的最快方法
    mask_upsampled = torch.repeat_interleave(mask_frames, repeats=frame_len, dim=1)
    
    # 8. 裁剪回原始长度
    # 去掉之前为了整除 padding 的部分
    mask_final = mask_upsampled[:, :L]
    
    return mask_final
class SinusoidalPositionalEncodingBase(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dim_t = embedding_dim

    def _get_sinusoidal_encoding(self, indices, dim):
        """
        根据给定的索引计算正弦位置编码
        indices: [N] 
        dim: 编码维度
        """
        device = indices.device
        # indices 形状变为 [N, 1]
        pos = indices.float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim)
        )
        pe = torch.zeros(indices.size(0), dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe
class quen_whisper_Encoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.position_dim=48
        self.whisper_encoder = whisper(**kwargs)
        self.quen_encoder = quen(**kwargs)
        self.position_encoder = SinusoidalPositionalEncodingBase(self.position_dim)
        self.output_dim = self.quen_encoder.output_dim+self.whisper_encoder.output_dim+self.position_dim

    def forward(self, input_features, attention_mask=None):
        valid_mask=get_vad_mask(input_features)
        whisper_outputs,whisper_mask = self.whisper_encoder(audio=input_features, audio_attention_mask=attention_mask,audio_valid_mask =valid_mask)
        quen_outputs,quen_mask = self.quen_encoder(input_features, attention_mask)
        # 拼接两个编码器的输出
        max_length = max(whisper_outputs.size(1), quen_outputs.size(1))
        if whisper_outputs.size(1) < max_length:
            pad_size = max_length - whisper_outputs.size(1)
            pad_tensor = torch.zeros(whisper_outputs.size(0), pad_size, whisper_outputs.size(2), device=whisper_outputs.device)
            whisper_outputs = torch.cat((whisper_outputs, pad_tensor), dim=1)
            whisper_mask = torch.cat((whisper_mask, torch.zeros(whisper_mask.size(0), pad_size, device=whisper_mask.device)), dim=1)
        elif quen_outputs.size(1) < max_length:
            pad_size = max_length - quen_outputs.size(1)
            pad_tensor = torch.zeros(quen_outputs.size(0), pad_size, quen_outputs.size(2), device=quen_outputs.device)
            quen_outputs = torch.cat((quen_outputs, pad_tensor), dim=1)
            quen_mask = torch.cat((quen_mask, torch.zeros(quen_mask.size(0), pad_size, device=quen_mask.device)), dim=1)
        position_embeddings = self.position_encoder._get_sinusoidal_encoding(torch.arange(whisper_outputs.size(1), device=whisper_outputs.device),dim=self.position_dim)
        position_embeddings=position_embeddings.unsqueeze(0).expand(whisper_outputs.size(0),-1,-1)
        combined_outputs = torch.cat((whisper_outputs, quen_outputs,position_embeddings), dim=-1)

        #change mask type
        whisper_mask=whisper_mask.bool().to(position_embeddings.device)
        quen_mask=quen_mask.bool().to(position_embeddings.device)
        mask=whisper_mask & quen_mask
        return combined_outputs, mask
if __name__ == "__main__":

    import sys

    import os

    # from .quen import QwenEncoder as quen
    # from .whisper_base import Encoder as whisper
    from example.Quen2_whisper_silence.quen import QwenEncoder as quen
    from example.Quen2_whisper_silence.whisper_base import Encoder as whisper
    import torch
    enc = quen_whisper_Encoder()

    from src.xares_llm.audio_encoder_checker import check_audio_encoder
    check_audio_encoder(enc)  
