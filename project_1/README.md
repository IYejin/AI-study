# project_1 (CIFAR-10 CNN from scratch)

- Dataset: CIFAR-10 (32x32 RGB, 10 classes)
- Framework: PyTorch

## Run (Colab)
```bash
pip install -r project_1/requirements.txt
python -m project_1.src.train --model resnet18 --epochs 100 --batch_size 128
python -m project_1.src.eval --ckpt project_1/checkpoints/best.pt

