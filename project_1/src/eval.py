import argparse
import torch
import torch.nn as nn

from .data import get_dataloaders
from .models import SimpleCNN, resnet18_cifar

def get_model(name: str):
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN(num_classes=10)
    if name == "resnet18":
        return resnet18_cifar(num_classes=10)
    raise ValueError(f"Unknown model: {name}")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)

    ckpt = torch.load(args.ckpt, map_location=device)
    model_name = ckpt["model_name"]
    model = get_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += x.size(0)

    print(f"model={model_name} | test_loss={total_loss/total:.4f} | test_acc={total_correct/total:.4f}")

if __name__ == "__main__":
    main()
