import argparse
import ctypes
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from model import FaceClassifier


def prevent_sleep():
    """Prevent Windows from entering sleep mode during long training runs."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("Sleep prevention mode ACTIVE.")


def force_sleep():
    """Restore sleep policy and suspend Windows after training finishes."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        print("Training finished. Putting system to sleep...")
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 binary classifier for Real vs. AI-Generated faces."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Data_cropped",
        help="Root directory containing 'fake_faces' and 'real_faces' subfolders.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument(
        "--output",
        type=str,
        default="best_model.pth",
        help="Path to save the best validation checkpoint.",
    )
    parser.add_argument(
        "--sleep-after-train",
        action="store_true",
        help="Put Windows to sleep automatically after training completes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prevent_sleep()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")

    # Training augmentations include GaussianBlur (p=0.25) to prevent the model
    # from relying on image sharpness as a shortcut between real and AI faces.
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=train_transform)
    print(f"Classes (index -> name): {full_dataset.classes}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Split dataset: {train_size} training images | {val_size} validation images.")

    oversample_reply = input(
        "\nIncrease sampling probability of collage (Grok/GPT/Gemini/Ideogram) faces by 3x? (y/n): "
    )
    if oversample_reply.strip().lower() in ("y", "yes"):
        print("WeightedRandomSampler (3x oversampling for collage faces) enabled.")
        weights = []
        for idx in train_dataset.indices:
            path, _ = full_dataset.imgs[idx]
            w = 3.0 if "collage_" in os.path.basename(path) else 1.0
            weights.append(w)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("Initializing model...")
    model = FaceClassifier(num_classes=2).to(device)

    # Dynamic class weighting to compensate for Real > Fake imbalance
    fake_path = os.path.join(args.data_dir, "fake_faces")
    real_path = os.path.join(args.data_dir, "real_faces")
    num_fake = len(os.listdir(fake_path)) if os.path.exists(fake_path) else 1
    num_real = len(os.listdir(real_path)) if os.path.exists(real_path) else 1

    ratio = num_real / max(1, num_fake)
    class_weights = torch.tensor([ratio, 1.0], dtype=torch.float).to(device)
    print(f"Dataset distribution: {num_fake} Fake | {num_real} Real")
    print(f"Applied loss class weights: Fake={ratio:.2f}, Real=1.00")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists("checkpoint.pth"):
        resume_reply = input("\nFound existing 'checkpoint.pth'. Resume from last epoch? (y/n): ")
        if resume_reply.strip().lower() in ("y", "yes"):
            checkpoint = torch.load("checkpoint.pth", map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            print(f"Resuming training from epoch {start_epoch + 1}...\n")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n================ EPOCH {epoch + 1}/{args.epochs} ================")

        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc="Train", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc="Val  ", leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels).item()
                val_loss += batch_loss
                val_bar.set_postfix(loss=batch_loss)

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.output)
            print(f"New best model saved to '{args.output}'!")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            "checkpoint.pth",
        )
        print("Checkpoint updated ('checkpoint.pth'). You may safely stop with Ctrl+C.")

    if args.sleep_after_train:
        force_sleep()


if __name__ == "__main__":
    main()
