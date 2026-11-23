import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


def crop_folder(mtcnn, src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(src_dir):
        print(f"[!] Source directory not found: {src_dir}")
        return

    files = sorted(os.listdir(src_dir))
    for img_name in tqdm(files, desc=f"Cropping {src_dir}"):
        img_path = os.path.join(src_dir, img_name)
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
                    cropped = img.crop((left, top, right, bottom))
                    cropped.save(os.path.join(dst_dir, img_name))
        except Exception as exc:
            print(f"[!] Failed to process {img_name}: {exc}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mtcnn = MTCNN(keep_all=False, device=device)

    crop_folder(mtcnn, os.path.join("Data", "fake_faces"), os.path.join("Data_cropped", "fake_faces"))
    crop_folder(mtcnn, os.path.join("Data", "real_faces"), os.path.join("Data_cropped", "real_faces"))


if __name__ == "__main__":
    main()
