import torch
import os
from transformers import WhisperModel, WhisperProcessor
import torch.nn.functional as F
def length_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = lengths.amax()
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)
    return mask.long()

class Encoder(torch.nn.Module):
    def __init__(self, model_name="openai/whisper-large", local_dir=None):
        super().__init__()
        self.local_dir = local_dir
        self.model_name = model_name
        
        # 创建本地目录
        # os.makedirs(local_dir, exist_ok=True)
        # model_path = self.local_dir
        # 检查本地是否已有模型文件
        #model_path = os.path.join(local_dir, model_name.replace("/", "--"))
        
        if self.local_dir is not None and self._check_local_model_exists(self.local_dir):
            print(f"从本地加载模型: {self.local_dir}")
            self.processor = WhisperProcessor.from_pretrained(self.local_dir)
            self.model = WhisperModel.from_pretrained(self.local_dir).get_encoder()
        else:
            print(f"从网络下载模型: {model_name}")
            # 从网络下载
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperModel.from_pretrained(model_name).get_encoder()
            
            # 保存到本地
            # self._save_model_to_local(self.local_dir)
        
        self.output_dim = self.model.config.d_model

    def _check_local_model_exists(self, model_path):
        """检查本地模型文件是否存在"""
        required_files = [
            "config.json", 
            "tokenizer.json"
        ]
        
        if not os.path.exists(model_path):
            return False
            
        # 检查关键文件是否存在
        existing_files = os.listdir(model_path)
        for file in required_files:
            if file not in existing_files:
                return False
                
        return True

    def _save_model_to_local(self, model_path):
        """将模型保存到本地"""
        print(f"保存模型到本地: {model_path}")
        
        # 保存processor
        self.processor.save_pretrained(model_path)
        
        # 保存模型（需要保存整个WhisperModel，而不仅仅是encoder）
        # 因为get_encoder()返回的是模型的一部分，我们需要保存完整的模型
        full_model = WhisperModel.from_pretrained(self.model_name)
        full_model.save_pretrained(model_path)
        
        print("模型保存完成")

    def forward(self, audio: torch.Tensor, audio_attention_mask=None,audio_valid_mask = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Since feature extraction is on cpu this is super slow
        assert isinstance(audio, torch.Tensor)
        audio = audio.cpu().numpy()
        if audio.ndim == 1:
            # Single audio sequence
            audio_list = [audio]
        elif audio.ndim == 2:
            # Batch of audio sequences
            audio_list = [a for a in audio]
        else:
            raise ValueError("Audio tensor must be 1D (single sequence) or 2D (batch of sequences).")
        if audio_attention_mask is None:
            audio_lens = torch.tensor([a.shape[-1] for a in audio_list])
        else:
            audio_lens = audio_attention_mask.sum(-1)

        mel_lengths = audio_lens // self.processor.feature_extractor.hop_length
        feature_lengths = (mel_lengths - 1) // 2 + 1
        feature_lengths = (feature_lengths - 1) // 2 + 1
        trim_length = feature_lengths.amax()
        attention_mask = length_to_mask(feature_lengths)

        if audio_valid_mask is not None:
            audio_attention_mask=attention_mask
        #
        mask_float = audio_valid_mask.float().unsqueeze(1)
        
        # 2. 使用平均池化下采样 4 倍
        # kernel_size=4, stride=4 保证了无重叠的 4 倍压缩
        # ceil_mode=False (默认) 会丢弃末尾不足 4 个采样点的部分，与特征提取器行为对齐
        downsampled_float = F.avg_pool1d(mask_float, kernel_size=self.processor.feature_extractor.hop_length*4, stride=self.processor.feature_extractor.hop_length*4)
        
        # 3. 阈值判断并转回 bool
        # 如果 4 个点中至少有 3 个点有效 (即 > 0.5)，则该特征帧设为 True
        # 你也可以根据需求调整阈值，例如只要有 1 个点有效就设为 True ( > 0 )
        downsampled_mask = (downsampled_float.squeeze(1) > 0.5)
        if downsampled_mask.size(1)<trim_length:#append 1
            downsampled_mask=torch.cat((downsampled_mask, torch.ones(downsampled_mask.size(0), trim_length - downsampled_mask.size(1), device=downsampled_mask.device, dtype=torch.bool)), dim=1)
        downsampled_mask=downsampled_mask[:,:trim_length]

        attention_mask = attention_mask & downsampled_mask

        #


        features = self.processor(audio_list, sampling_rate=16000, return_tensors="pt").to(self.model.device)
        output = self.model(**features).last_hidden_state
        return output[:, :trim_length, :], attention_mask


if __name__ == "__main__":
    # 第一次运行：从网络下载并保存到本地
    enc = Encoder(local_dir="/asdata/lmd/whisper_large")
    valid_mask=torch.tensor([[1]*2000, [1]*400 + [0]*1600 ,  [0]*2000, [1]*1300 + [0]*700])
    q, p = enc(torch.randn(4, 2000), length_to_mask(torch.tensor([2000, 2000,2000, 2000])),valid_mask)
    print(q.shape, enc.output_dim)
    print(p.shape)
    print(p)
    
# import torch
# from transformers import WhisperModel, WhisperProcessor


# def length_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
#     if max_len is None:
#         max_len = lengths.amax()
#     idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
#     mask = idx < lengths.unsqueeze(1)
#     return mask.long()


# class WhisperEncoder(torch.nn.Module):
#     def __init__(self, model_name="openai/whisper-base"):
#         super().__init__()
#         self.processor = WhisperProcessor.from_pretrained(model_name)
#         self.model = WhisperModel.from_pretrained(model_name).get_encoder()
#         self.output_dim = self.model.config.d_model

#     def forward(self, audio: torch.Tensor, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
#         # Since feature extraction is on cpu this is super slow
#         assert isinstance(audio, torch.Tensor)
#         audio = audio.cpu().numpy()
#         if audio.ndim == 1:
#             # Single audio sequence
#             audio_list = [audio]
#         elif audio.ndim == 2:
#             # Batch of audio sequences
#             audio_list = [a for a in audio]
#         else:
#             raise ValueError("Audio tensor must be 1D (single sequence) or 2D (batch of sequences).")
#         if audio_attention_mask is None:
#             audio_lens = torch.tensor([a.shape[-1] for a in audio_list])
#         else:
#             audio_lens = audio_attention_mask.sum(-1)
#         mel_lengths = audio_lens // self.processor.feature_extractor.hop_length
#         feature_lengths = (mel_lengths - 1) // 2 + 1
#         feature_lengths = (feature_lengths - 1) // 2 + 1
#         trim_length = feature_lengths.amax()
#         attention_mask = length_to_mask(feature_lengths)

#         features = self.processor(audio_list, sampling_rate=16000, return_tensors="pt").to(self.model.device)
#         output = self.model(**features).last_hidden_state
#         return output[:, :trim_length, :], attention_mask


# if __name__ == "__main__":
#     enc = WhisperEncoder()
#     q, _ = enc(torch.randn(4, 16000), length_to_mask(torch.tensor([16000, 8000, 4000, 2000])))
#     print(q.shape, enc.output_dim)
