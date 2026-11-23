import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


def main():
    print("=====================================================")
    print(" MTCNN CELEBA-HQ FACE EXTRACTOR")
    print("=====================================================")

    input_dir = "celeba_hq_256"
    output_dir = os.path.join("Data_cropped", "real_faces")

    if not os.path.exists(input_dir):
        print(f"[!] Input directory '{input_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    # Cap extraction at 10,000 images to keep real/fake class ratio manageable
    max_images = 10000
    image_files = image_files[:max_images]

    if not image_files:
        print(f"\n[!] No images found in '{input_dir}'.")
        return

    print(f"\n[+] Processing {len(image_files)} CelebA-HQ images...\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    mtcnn = MTCNN(keep_all=False, device=device)
    face_counter = 1

    for img_file in tqdm(image_files, desc="Cropping CelebA-HQ faces"):
        img_path = os.path.join(input_dir, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            boxes, probs = mtcnn.detect(img)

            if boxes is not None and probs[0] > 0.90:
                left = max(0, int(boxes[0][0]))
                top = max(0, int(boxes[0][1]))
                right = min(img.width, int(boxes[0][2]))
                bottom = min(img.height, int(boxes[0][3]))

                if right > left and bottom > top:
                    face = img.crop((left, top, right, bottom))
                    save_path = os.path.join(output_dir, f"celeba_{face_counter}.jpg")
                    face.save(save_path, "JPEG", quality=95)
                    face_counter += 1
        except Exception:
            continue

    print("\n=====================================================")
    print("[*] COMPLETED!")
    print(f"Extracted {face_counter - 1} faces into '{output_dir}'.")
    print("=====================================================")


if __name__ == "__main__":
    main()
