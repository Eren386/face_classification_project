import argparse
import os
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision import transforms

from model import FaceClassifier


def evaluate_model(model_obj, device, mtcnn, val_transform, test_dir):
    y_true = []
    y_pred = []

    for label_dir, true_label in [("fake", 0), ("real", 1)]:
        dir_path = os.path.join(test_dir, label_dir)
        if not os.path.exists(dir_path):
            continue

        for img_name in sorted(os.listdir(dir_path)):
            img_path = os.path.join(dir_path, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            boxes, _ = mtcnn.detect(image)
            if boxes is None:
                continue

            box = boxes[0]
            left = max(0, int(box[0]))
            top = max(0, int(box[1]))
            right = min(image.width, int(box[2]))
            bottom = min(image.height, int(box[3]))
            cropped_image = image.crop((left, top, right, bottom))

            input_tensor = val_transform(cropped_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model_obj(input_tensor)
                probs = F.softmax(output, dim=1)
                _, pred = torch.max(probs, 1)

            y_true.append(true_label)
            y_pred.append(pred.item())

    if len(y_true) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two FaceClassifier checkpoints head-to-head."
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default=os.path.join("weights", "Face_Detector_v1_Gold.pth"),
        help="Path to the first checkpoint (baseline).",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="best_model.pth",
        help="Path to the second checkpoint (candidate).",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="test_data",
        help="Directory containing 'fake/' and 'real/' subfolders.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_a = FaceClassifier(num_classes=2).to(device)
    model_a.load_state_dict(torch.load(args.model_a, map_location=device))
    model_a.eval()

    model_b = FaceClassifier(num_classes=2).to(device)
    model_b.load_state_dict(torch.load(args.model_b, map_location=device))
    model_b.eval()

    mtcnn = MTCNN(keep_all=False, device=device)
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print(f"Evaluating Model A ({args.model_a})...")
    metrics_a = evaluate_model(model_a, device, mtcnn, val_transform, args.test_dir)

    print(f"Evaluating Model B ({args.model_b})...")
    metrics_b = evaluate_model(model_b, device, mtcnn, val_transform, args.test_dir)

    print("\n" + "=" * 64)
    print(" MODEL COMPARISON REPORT")
    print("=" * 64)
    print(f"{'Metric':<15} | {'Model A (Baseline)':<20} | {'Model B (Candidate)':<20}")
    print("-" * 64)
    print(
        f"{'Accuracy':<15} | {metrics_a['accuracy']*100:<19.2f}% | {metrics_b['accuracy']*100:<19.2f}%"
    )
    print(
        f"{'Precision':<15} | {metrics_a['precision']*100:<19.2f}% | {metrics_b['precision']*100:<19.2f}%"
    )
    print(
        f"{'Recall':<15} | {metrics_a['recall']*100:<19.2f}% | {metrics_b['recall']*100:<19.2f}%"
    )
    print(f"{'F1 Score':<15} | {metrics_a['f1']:<20.4f} | {metrics_b['f1']:<20.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
