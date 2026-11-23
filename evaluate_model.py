import argparse
import os
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torchvision import transforms

from model import FaceClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FaceClassifier checkpoint on a real/fake test directory."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("weights", "Face_Detector_v1_Gold.pth"),
        help="Path to the .pth model weights file.",
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
    print(f"Loading model: {args.model}")

    model = FaceClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    mtcnn = MTCNN(keep_all=False, device=device)

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    y_true = []
    y_pred = []
    results = []

    generator_stats = {
        "gemini": {"total": 0, "correct": 0},
        "gpt": {"total": 0, "correct": 0},
        "grok": {"total": 0, "correct": 0},
        "ideogram": {"total": 0, "correct": 0},
        "tpdne": {"total": 0, "correct": 0},
        "other": {"total": 0, "correct": 0},
    }

    for label_dir, true_label in [("fake", 0), ("real", 1)]:
        dir_path = os.path.join(args.test_dir, label_dir)
        if not os.path.exists(dir_path):
            print(f"[!] Warning: test subfolder not found: {dir_path}")
            continue

        for img_name in sorted(os.listdir(dir_path)):
            img_path = os.path.join(dir_path, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            boxes, _ = mtcnn.detect(image)
            if boxes is None:
                results.append(
                    {
                        "name": img_name,
                        "true_label": "Real" if true_label == 1 else "Fake",
                        "pred_label": "NO FACE DETECTED",
                        "confidence": 0.0,
                        "is_correct": False,
                    }
                )
                continue

            box = boxes[0]
            left = max(0, int(box[0]))
            top = max(0, int(box[1]))
            right = min(image.width, int(box[2]))
            bottom = min(image.height, int(box[3]))
            cropped_image = image.crop((left, top, right, bottom))

            input_tensor = val_transform(cropped_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)

            pred_val = pred.item()
            conf_val = confidence.item() * 100
            is_correct = pred_val == true_label

            y_true.append(true_label)
            y_pred.append(pred_val)

            results.append(
                {
                    "name": img_name,
                    "true_label": "Real" if true_label == 1 else "Fake",
                    "pred_label": "Real" if pred_val == 1 else "Fake",
                    "confidence": conf_val,
                    "is_correct": is_correct,
                }
            )

            if true_label == 0:
                lower_name = img_name.lower()
                cat = "other"
                if "gemini" in lower_name:
                    cat = "gemini"
                elif "gpt" in lower_name or "gtp" in lower_name:
                    cat = "gpt"
                elif "grok" in lower_name:
                    cat = "grok"
                elif "ideogram" in lower_name:
                    cat = "ideogram"
                elif "tpdne" in lower_name:
                    cat = "tpdne"

                generator_stats[cat]["total"] += 1
                if is_correct:
                    generator_stats[cat]["correct"] += 1

    print("\n" + "=" * 54)
    print(" REPORT 1: PREDICTIONS SUMMARY (FAKE / REAL)")
    print("=" * 54)
    for r in results:
        print(f"[{r['true_label']}] {r['name']} -> Prediction: {r['pred_label']}")

    print("\n" + "=" * 54)
    print(" REPORT 2: CONFIDENCE SCORES")
    print("=" * 54)
    for r in results:
        status = "[CORRECT]" if r["is_correct"] else "[WRONG]  "
        if r["pred_label"] == "NO FACE DETECTED":
            print(f"{status} | {r['name']} -> NO FACE DETECTED")
        else:
            print(
                f"{status} | {r['name']} -> True: {r['true_label']} | "
                f"Pred: {r['pred_label']} ({r['confidence']:.2f}%)"
            )

    print("\n" + "=" * 54)
    print(" REPORT 3: BREAKDOWN BY AI GENERATOR")
    print("=" * 54)
    for gen, stats in generator_stats.items():
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            print(
                f"Generator: {gen.upper():<10} | Total: {stats['total']:<2} | "
                f"Correct: {stats['correct']:<2} | Accuracy: {acc:.2f}%"
            )

    if len(y_true) > 0:
        print("\n" + "=" * 54)
        print(" OVERALL METRICS & CONFUSION MATRIX")
        print("=" * 54)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        print("Confusion Matrix:")
        print("                 Pred Fake(0)    Pred Real(1)")
        print(f"True Fake(0)     {cm[0][0]:<15} {cm[0][1]}")
        print(f"True Real(1)     {cm[1][0]:<15} {cm[1][1]}")
        print("\nMetrics:")
        print(f"Accuracy:           {acc * 100:.2f}%")
        print(f"Recall (Real):      {rec * 100:.2f}%")
        print(f"F1 Score (Real):    {f1:.4f}")


if __name__ == "__main__":
    main()
