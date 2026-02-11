# The commit for The Interspeech 2026 Audio Encoder Capability Challenge from IASP Lab

1、 To enable the use of this encoder,you should create a conda environment with the following cmd command:

```cmd
conda create -n aec python=3.10.2

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
accelerate launch -m xares_llm.run quen_whisper.py task1 task1
```

