import argparse
import torch
import torch.nn as nn

from thop import profile, clever_format

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
def eval_test_acc(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch", type=int, default=1, help="dummy input batch for FLOPs (usually 1)")
    parser.add_argument("--no_acc", action="store_true", help="skip test accuracy eval")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    model_name = ckpt["model_name"]
    model = get_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # FLOPs / Params (CIFAR-10 input: 3x32x32)
    dummy = torch.randn(args.batch, 3, 32, 32).to(device)

    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops = 2 * macs  # 관례적으로 FLOPs = 2 * MACs

    print("RAW macs:", macs)
    print("RAW flops:", flops)
    print("RAW params:", params)


    macs_f, params_f = clever_format([macs, params], "%.3f")
    flops_f, = clever_format([flops], "%.3f")[0]

    print(f"Model: {model_name}")
    print(f"Params: {params_f}")
    print(f"MACs  : {macs_f}  (for batch={args.batch})")
    print(f"FLOPs : {flops_f} (≈ 2×MACs, for batch={args.batch})")
    if args.batch == 1:
        print("Note: batch=1이면 '이미지 1장당' FLOPs로 보면 됩니다.")

    # Accuracy (optional)
    if not args.no_acc:
        _, _, test_loader = get_dataloaders(data_dir=args.data_dir, batch_size=256)
        acc = eval_test_acc(model, test_loader, device)
        print(f"Test Acc: {acc:.4f}")

if __name__ == "__main__":
    main()
