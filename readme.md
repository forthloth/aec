# WQ-Fusion

## This is the official implementation of the paper WQ-Fusion for Interspeech 2026, the paper could be found at WQ-Fusion.pdf in the files.

1、 To enable the use of this encoder,you should create a conda environment with the following cmd command:

```cmd
conda create -n aec python=3.10.2
conda activate aec
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

pip install git+https://github.com/xiaomi-research/xares-llm
```

 2、 Clone the code

```cmd
git clone https://github.com/forthloth/aec.git
cd aec
```

3、Evaluate the encoder for the challenge with

```cmd
export HF_ENDPOINT=https://hf-mirror.com
accelerate launch -m xares_llm.run quen_whisper.py task1 task1
```



If your device doesn't support accelerate,try

```
export HF_ENDPOINT=https://hf-mirror.com
python -m xares_llm.run quen_whisper.py task1 task1
```

Instead.




The Evaluation results:

| Task                      | Dasheng-Base Score | Whisper-Base Score | Ours  |
| ------------------------- | ------------------ | ------------------ | ----- |
| eval_asvspoof2015         | 0.937              | 0.943              | 0.979 |
| eval_cremad               | 0.621              | 0.516              | 0.820 |
| eval_esc-50               | 0.755              | 0.635              | 0.930 |
| eval_fluentspeechcommands | 0.984              | 0.817              | 0.995 |
| eval_freemusicarchive     | 0.429              | 0.579              | 0.725 |
| eval_fsd50k               | 0.063              | 0.092              | 0.295 |
| eval_fsdkaggle2018        | 0.415              | 0.552              | 0.828 |
| eval_gtzan                | 0.323              | 0.697              | 0.929 |
| eval_libricount           | 0.386              | 0.409              | 0.583 |
| eval_nsynth               | 0.675              | 0.638              | 0.748 |
| eval_speechcommandsv1     | 0.655              | 0.694              | 0.938 |
| eval_urbansound8k         | 0.829              | 0.737              | 0.871 |
| eval_vocalsound           | 0.855              | 0.867              | 0.938 |
| eval_voxceleb1            | 0.974              | 0.762              | 0.985 |
| eval_voxlingua33          | 0.311              | 0.835              | 0.975 |
| Overall                   | 0.614              | 0.652              | 0.836 |

