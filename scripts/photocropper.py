import os
from PIL import Image


def crop_bottom(img: Image.Image, crop_height: int = 30) -> Image.Image:
    width, height = img.size
    if height <= crop_height:
        return img
    return img.crop((0, 0, width, height - crop_height))


def crop_fake_images(fake_folder_path: str, crop_height: int = 30, output_folder: str = None):
    if output_folder is None:
        output_folder = fake_folder_path + "_cropped"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(fake_folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(fake_folder_path, filename)
            img = Image.open(img_path)
            cropped_img = crop_bottom(img, crop_height)
            save_path = os.path.join(output_folder, filename)
            cropped_img.save(save_path)
            print(f"Cropped and saved: {save_path}")


if __name__ == "__main__":
    folder = "fake_faces"
    if os.path.exists(folder):
        crop_fake_images(folder, crop_height=30)
    else:
        print(f"Directory '{folder}' not found.")
