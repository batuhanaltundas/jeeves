

```bash
conda create -n voice-agent python=3.11 -y
conda activate voice-agent
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio
# verify:
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
pip install faster-whisper sounddevice scipy
pip install transformers accelerate safetensors sentencepiece
```


```bash
python test_qwen.py
```