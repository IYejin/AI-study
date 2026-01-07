import argparse
import time
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data import get_dataloaders
from .models import SimpleCNN, resnet18_cifar
from .utils import set_seed, accuracy_top1, ensure_dir

def get_model(name: str):
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN(num_classes=10)
    if name == "resnet18":
        return resnet18_cifar(num_classes=10)
    raise ValueError(f"Unknown model: {name}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_acc  += accuracy_top1(logits, y) * x.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="val"):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in tqdm(loader, desc=split_name, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_acc  += accuracy_top1(logits, y) * x.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_acc / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", choices=["simplecnn", "resnet18"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--ckpt_dir", type=str, default="project_1/checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2,
        val_ratio=0.1,
        seed=args.seed
    )

    model = get_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ensure_dir(args.ckpt_dir)
    best_val_acc = 0.0
    best_path = f"{args.ckpt_dir}/best.pt"

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device, split_name="val")
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"lr={lr_now:.4f} | "
              f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_name": args.model,
                "model_state": model.state_dict(),
                "val_acc": best_val_acc,
                "epoch": epoch,
                "args": vars(args),
            }, best_path)
            print(f"  ✅ Saved best checkpoint: {best_path} (val_acc={best_val_acc:.4f})")

    print("Training done. elapsed:", round(time.time() - t0, 1), "sec")

    # best checkpoint로 test 평가
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, split_name="test")
    print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f}")

if __name__ == "__main__":
    main()
