import math
from collections.abc import Mapping
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
)
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig
from transformers.models.qwen2_audio.processing_qwen2_audio import Qwen2AudioProcessorKwargs
import os
import pathlib
class quen_encoder(nn.Module):
    def __init__(self, config: Qwen2AudioConfig):
        super().__init__(config)
        self.audio_tower = AutoModel.from_config(config.audio_config)  # Usually a `Qwen2AudioEncoder` instance
    def forward(self, *args, **kwargs):
        return self.audio_tower(*args, **kwargs)
    

def length_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = lengths.amax()
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)
    return mask.long()
class QwenEncoder(nn.Module):
    """
    Fusion encoder:
      - shared waveform preprocessing (optional): RemoveSilence + RemoveDC + RMS Norm
      - Dasheng: pad/trunc -> encode -> pool to 25Hz length (T based on real length)
      - Qwen2-Audio: ONLY audio_tower (no 7B LLM), pad/trunc -> encode -> pool to 25Hz
      - concat -> projection to output_dim

    forward() returns:
      output: [B, T_25hz, output_dim]
      audio_attention_mask: passthrough
    """

    def __init__(
        self,
        qwen_name: str = "Qwen/Qwen2-Audio-7B-Instruct",quen_local=None,
        target_hz: int = 25,
        dasheng_pad_s: float = 10.0,
        qwen_pad_s: float = 30.0,
        do_remove_silence: bool = True,
        silence_thresh: float = 0.01,
        freeze_qwen: bool = True,
        qwen_torch_dtype: str = "float16",  
        **kwargs,
    ) -> None:
        super().__init__()


        self.target_hz = int(target_hz)

        self.dasheng_pad_s = float(dasheng_pad_s)
        self.qwen_pad_s = float(qwen_pad_s)

        self.do_remove_silence = bool(do_remove_silence)
        self.silence_thresh = float(silence_thresh)


        # Qwen2-Audio (AUDIO TOWER ONLY)
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_name, trust_remote_code=True)

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        qwen_dtype = dtype_map.get(str(qwen_torch_dtype).lower(), torch.float16)

        # Load full model on CPU (Trainer will move our module later).
        # parent_path = pathlib.Path(quen_local)
        # path = [d for d in parent_path.iterdir() if d.is_dir()]
        if quen_local is not None and os.path.exists(quen_local):
                qwen_full = Qwen2AudioForConditionalGeneration.from_pretrained(
                    quen_local,
                    trust_remote_code=True,
                    torch_dtype=qwen_dtype,
                    low_cpu_mem_usage=True,
                    device_map=None,
                )
        else:
            print(f"本地模型不存在，从网络下载: {qwen_name}")
            qwen_full = Qwen2AudioForConditionalGeneration.from_pretrained(
                qwen_name,
                trust_remote_code=True,
                torch_dtype=qwen_dtype,
                low_cpu_mem_usage=True,
                device_map=None,
            )
            # qwen_full.save_pretrained(quen_local)

        # Keep only audio tower
        if not hasattr(qwen_full, "audio_tower"):
            raise RuntimeError("Qwen2AudioForConditionalGeneration has no attribute audio_tower.")
        self.qwen_audio = qwen_full.audio_tower

        # Delete the big 7B text model / head to avoid .to(cuda) OOM
        if hasattr(qwen_full, "model"):
            del qwen_full.model
        if hasattr(qwen_full, "lm_head"):
            del qwen_full.lm_head
        del qwen_full

        # sampling rate
        fe = getattr(self.qwen_processor, "feature_extractor", None)
        self.sample_rate = getattr(fe, "sampling_rate", None)
        if self.sample_rate is None:
            self.sample_rate = getattr(self.dasheng_fe, "sampling_rate", 16000)

        # Infer qwen audio tower dim (robust)
        qwen_dim = None
        # Common locations
        qcfg = getattr(self.qwen_audio, "config", None)
        if qcfg is not None:
            for key in ["hidden_size", "d_model", "encoder_embed_dim", "model_dim"]:
                if hasattr(qcfg, key):
                    qwen_dim = getattr(qcfg, key)
                    break
            if qwen_dim is None and isinstance(qcfg, dict):
                for key in ["hidden_size", "d_model", "encoder_embed_dim", "model_dim"]:
                    if key in qcfg:
                        qwen_dim = qcfg[key]
                        break
        if qwen_dim is None:
            raise RuntimeError("Cannot infer Qwen2-AudioTower output dim.")

        print(f"[Qwen2-AudioTower] inferred dim = {qwen_dim}")

        self.output_dim =int(qwen_dim)
        # Freeze bases
 

        if freeze_qwen:
            for p in self.qwen_audio.parameters():
                p.requires_grad = False
            self.qwen_audio.eval()
    def _check_local_model_exists(self, model_path):
        """检查本地模型文件是否存在"""

        parent_path = pathlib.Path( model_path)
        path = [d for d in parent_path.iterdir() if d.is_dir()]
        if len(path) == 1:
                single_subdir = path[0]
                # single_subdir 已经是完整路径对象
                print(f"确认成功！子文件夹名称: {single_subdir.name}")
                single_subdir=str(single_subdir)
        else:
                raise ValueError(f"预期有1个子文件夹，实际找到 {len(path)} 个。")
        model_path=single_subdir
        required_files = [
            "config.json", 
            "model.safetensors",  # 或 model.safetensors
            "preprocessor_config.json"
        ]
        
        if not os.path.exists(model_path):
            return False
            
        # 检查关键文件是否存在
        existing_files = os.listdir(model_path)
        for file in required_files:
            if file not in existing_files:
                return False
                
        return True

    def _remove_dc_and_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean()
        rms = torch.sqrt(torch.mean(x * x) + 1e-8)
        x = x / (rms + 1e-8)
        return x

    def _remove_silence(self, x: torch.Tensor) -> torch.Tensor:
        if not self.do_remove_silence:
            return x
        mask = (x.abs() > self.silence_thresh)
        if not mask.any():
            return x
        idx = mask.nonzero(as_tuple=False).view(-1)
        s = int(idx[0].item())
        e = int(idx[-1].item()) + 1
        return x[s:e]

    def _preprocess_batch(self, audio: torch.Tensor) -> Tuple[List[torch.Tensor], List[int]]:
        """
        audio: [B, L] waveform
        Returns:
          xs: list of 1D tensors on CPU
          lens: real lengths (after silence trim)
        """
        xs: List[torch.Tensor] = []
        lens: List[int] = []

        audio_cpu = audio.detach().cpu()
        for b in range(audio_cpu.size(0)):
            x = audio_cpu[b]
            x = self._remove_silence(x)
            x = self._remove_dc_and_norm(x)
            xs.append(x)
            lens.append(int(x.numel()))
        return xs, lens

    def _pad_or_trunc(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        L = x.numel()
        if L == target_len:
            return x
        if L > target_len:
            return x[:target_len]
        pad = x.new_zeros(target_len - L)
        return torch.cat([x, pad], dim=0)

    @staticmethod
    def _pool_to_T(x: torch.Tensor, T: int) -> torch.Tensor:
        """
        x: [B, T0, D] -> [B, T, D]
        """
        if x.size(1) == T:
            return x
        x_ = x.transpose(1, 2)               # [B, D, T0]
        x_ = F.adaptive_avg_pool1d(x_, T)    # [B, D, T]
        return x_.transpose(1, 2)            # [B, T, D]

    def _seconds_to_T25(self, n_samples: int) -> int:
        dur_s = n_samples / float(self.sample_rate)
        return max(1, int(math.ceil(dur_s * self.target_hz)))

    # Helpers for Qwen processor / audio tower
    @staticmethod
    def _extract_audio_features_from_processor(proc_out: Mapping) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Qwen/Whisper-like processor may return:
          - input_features (+ attention_mask)
          - or input_values (+ attention_mask)
        """
        if "input_features" in proc_out:
            feats = proc_out["input_features"]
            attn = proc_out.get("attention_mask", None)
            return feats, attn
        if "input_values" in proc_out:
            feats = proc_out["input_values"]
            attn = proc_out.get("attention_mask", None)
            return feats, attn
        raise RuntimeError(f"Unexpected processor output keys: {list(proc_out.keys())}")

    @staticmethod
    def _call_audio_tower(audio_tower: nn.Module, feats: torch.Tensor, attn: Optional[torch.Tensor]):
        # Different versions may use different signatures; try robustly.
        try:
            if attn is not None:
                return audio_tower(feats, attention_mask=attn)
            return audio_tower(feats)
        except TypeError:
            # Some models might require named arg like input_features=
            try:
                if attn is not None:
                    return audio_tower(input_features=feats, attention_mask=attn)
                return audio_tower(input_features=feats)
            except TypeError:
                # Last resort: ignore mask
                return audio_tower(feats)

    @staticmethod
    def _extract_hidden(out_a) -> torch.Tensor:
        """
        Extract [B, T, D] hidden states from audio tower output.
        """
        if hasattr(out_a, "last_hidden_state") and out_a.last_hidden_state is not None:
            return out_a.last_hidden_state
        if isinstance(out_a, (tuple, list)) and len(out_a) > 0 and isinstance(out_a[0], torch.Tensor):
            return out_a[0]
        if isinstance(out_a, torch.Tensor):
            return out_a
        raise RuntimeError(f"Cannot extract hidden states from audio_tower output type: {type(out_a)}")

    # Dasheng forward
    def _forward_dasheng(self, xs: List[torch.Tensor], real_lens: List[int], device: torch.device) -> torch.Tensor:
        pad_len = int(round(self.dasheng_pad_s * self.sample_rate))

        xs_pad = [self._pad_or_trunc(x, pad_len) for x in xs]
        xs_np = [x.detach().cpu().float().numpy() for x in xs_pad]

        fe_out = self.dasheng_fe(
            xs_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )

        if isinstance(fe_out, Mapping):
            if "input_values" in fe_out:
                input_values = fe_out["input_values"]
            elif "input_features" in fe_out:
                input_values = fe_out["input_features"]
            else:
                raise RuntimeError(f"Dasheng feature output keys unexpected: {list(fe_out.keys())}")
        else:
            input_values = fe_out

        if not isinstance(input_values, torch.Tensor):
            raise RuntimeError(f"Dasheng input_values must be Tensor, got {type(input_values)}")

        input_values = input_values.to(device)

        out = self.dasheng(input_values=input_values)

        # Robust hidden extraction: hidden_states or last_hidden_state
        hs = getattr(out, "hidden_states", None)
        if hs is None:
            hs = getattr(out, "last_hidden_state", None)

        if hs is None:
            # Sometimes out itself could be a tensor
            if isinstance(out, torch.Tensor):
                hs = out
            else:
                raise RuntimeError("Dasheng forward returned no hidden_states/last_hidden_state.")

        # Align to T@25Hz based on real length (after trim)
        T_real = max(self._seconds_to_T25(L) for L in real_lens)
        hs_25 = self._pool_to_T(hs, T_real)
        return hs_25


    # Qwen audio_tower forward
    def _forward_qwen_audio(self, xs: List[torch.Tensor],  device: torch.device) -> torch.Tensor:
        # pad_len = int(round(self.qwen_pad_s * self.sample_rate))

        # xs_pad = [self._pad_or_trunc(x, pad_len) for x in xs]
        # xs_np = [x.detach().cpu().float().numpy() for x in xs_pad]
        xs_np = [x.detach().cpu().float().numpy() for x in xs]
        # output_kwargs = self.qwen_processor._merge_kwargs(
        #             Qwen2AudioProcessorKwargs,
        #             tokenizer_init_kwargs=self.qwen_processor.tokenizer.init_kwargs
        #         )
        # output_kwargs["audio_kwargs"]["return_attention_mask"] = True
        # output_kwargs["audio_kwargs"]["padding"] = "max_length"
        #a_in = self.qwen_processor.feature_extractor(audio=xs_np,raw_speech=True,**output_kwargs["audio_kwargs"])
        #get text input as the same long as audio input with 'good' token
        conversations = ['<|AUDIO|>']*len(xs_np)

        a_in = self.qwen_processor(audio=xs_np,text=conversations)      
        input_features, feature_attention_mask=a_in['input_features'],a_in['feature_attention_mask']
        input_features = torch.tensor(input_features)
        feature_attention_mask = torch.tensor(feature_attention_mask)
        #================================
        if input_features is not None :
                audio_feat_lengths, audio_output_lengths = self.qwen_audio._get_feat_extract_output_lengths(
                    feature_attention_mask.sum(-1)
                )
                batch_size, _, max_mel_seq_len = input_features.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                # Create a sequence tensor of shape (batch_size, max_seq_len)
                seq_range = (
                    torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
                # Create mask
                padding_mask = seq_range >= lengths_expand

                audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                    batch_size, 1, max_seq_len, max_seq_len
                )
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.qwen_audio.conv1.weight.dtype, device=self.qwen_audio.conv1.weight.device
                )
                audio_attention_mask[audio_attention_mask_] = float("-inf")
        #================================


        audio_outputs=self.qwen_audio(input_features=input_features.to(device),attention_mask=audio_attention_mask.to(device))
        audio_features= audio_outputs.last_hidden_state

        max_length=audio_output_lengths.max()
        audio_features=audio_features[:,:max_length,:]
        return audio_features,audio_output_lengths
        # hs = self._extract_hidden(out_a)
        # T_real = max(self._seconds_to_T25(L) for L in real_lens)
        # hs_25 = self._pool_to_T(hs, T_real)
        # return hs_25

    # public forward
    def forward(self, audio, audio_attention_mask=None):
        """
        audio: [B, L] waveform on same device as model parameters (Trainer will handle .to()).
        """
        device = next(self.parameters()).device
        # audio, real_lens = self._preprocess_batch(audio)
        out, audio_output_lengths = self._forward_qwen_audio(audio,  device)       # [B, T, D2]
        audio_attention_mask = length_to_mask(audio_output_lengths)
        #change out type to be float
        out = out.float()
                           
        return out, audio_attention_mask
if __name__ == "__main__":

    import sys
    import os
    #set cuda 6
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    enc = QwenEncoder()
    sys.path.append("/home/bms34/data/lmd/xares-llm-main")
    from src.xares_llm.audio_encoder_checker import check_audio_encoder
    check_audio_encoder(enc)
