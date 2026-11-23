import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mtcnn = MTCNN(keep_all=False, device=device)

    base_dir = "Data_cropped"
    target_dir = os.path.join(base_dir, "fake_faces")
    os.makedirs(target_dir, exist_ok=True)

    folders_to_process = ["1024", "512", "768"]

    image_paths = []
    for res_folder in folders_to_process:
        res_path = res_folder if os.path.exists(res_folder) else os.path.join(base_dir, res_folder)
        if not os.path.exists(res_path):
            continue

        for gender in ["man", "woman"]:
            gender_path = os.path.join(res_path, gender)
            if not os.path.exists(gender_path):
                continue

            for img_name in os.listdir(gender_path):
                img_path = os.path.join(gender_path, img_name)
                if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append((img_path, res_folder, gender, img_name))

    print(f"Total images to process: {len(image_paths)}")

    success_count = 0
    fail_count = 0

    for img_path, res_folder, gender, img_name in tqdm(image_paths, desc="Cropping Stable Diffusion faces"):
        try:
            img = Image.open(img_path).convert("RGB")
            boxes, _ = mtcnn.detect(img)

            if boxes is not None:
                box = boxes[0]
                left = max(0, int(box[0]))
                top = max(0, int(box[1]))
                right = min(img.width, int(box[2]))
                bottom = min(img.height, int(box[3]))

                if right > left and bottom > top:
                    face = img.crop((left, top, right, bottom))
                    new_name = f"diffusion_{res_folder}_{gender}_{img_name}"
                    save_path = os.path.join(target_dir, new_name)
                    if not save_path.lower().endswith(".jpg"):
                        save_path = save_path.rsplit(".", 1)[0] + ".jpg"
                    face.save(save_path)
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1
        except Exception:
            fail_count += 1
            continue

    print("\nProcessing complete!")
    print(f"Successfully cropped faces: {success_count}")
    print(f"No face detected / failed: {fail_count}")


if __name__ == "__main__":
    main()
