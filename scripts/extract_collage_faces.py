import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


def main():
    print("=====================================================")
    print(" MTCNN COLLAGE FACE EXTRACTOR")
    print("=====================================================")

    input_dir = os.path.join("Data", "raw_collages")
    output_dir = os.path.join("Data_cropped", "gpt_midjourney_fakes")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )

    if not image_files:
        print(f"\n[!] No collage images found in '{input_dir}'.")
        print("Place multi-face AI collage images in that folder and re-run this script.")
        return

    print(f"\n[+] Found {len(image_files)} collage images. Extracting faces...\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # keep_all=True extracts every face in multi-subject collages
    # min_face_size=30 allows catching smaller grid tiles
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=30)

    face_counter = 1

    for img_file in tqdm(image_files, desc="Processing collages"):
        img_path = os.path.join(input_dir, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            boxes, probs = mtcnn.detect(img)

            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob < 0.90:
                        continue

                    left = max(0, int(box[0]))
                    top = max(0, int(box[1]))
                    right = min(img.width, int(box[2]))
                    bottom = min(img.height, int(box[3]))

                    if right - left <= 0 or bottom - top <= 0:
                        continue

                    face = img.crop((left, top, right, bottom))
                    save_path = os.path.join(output_dir, f"collage_fake_{face_counter}.jpg")
                    face.save(save_path, "JPEG", quality=95)
                    face_counter += 1
        except Exception as exc:
            print(f"\n[!] Error processing {img_file}: {exc}")

    print("\n=====================================================")
    print("[*] EXTRACTION COMPLETE!")
    print(f"Successfully extracted {face_counter - 1} faces.")
    print(f"Saved to '{output_dir}'. Copy them into 'Data_cropped/fake_faces' before training.")
    print("=====================================================")


if __name__ == "__main__":
    main()
