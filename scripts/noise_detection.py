import os
import cv2
import numpy as np
from scipy.stats import entropy
from skimage.util import view_as_windows
from tqdm import tqdm


def calculate_entropy(image_gray: np.ndarray, window_size: int = 32) -> float:
    windows = view_as_windows(image_gray, (window_size, window_size), step=window_size)
    total_entropy = 0.0
    count = 0

    for row in windows:
        for patch in row:
            hist, _ = np.histogram(patch, bins=256, range=(0, 256), density=True)
            total_entropy += float(entropy(hist))
            count += 1

    return total_entropy / max(1, count)


def is_noisy(image_path: str, entropy_threshold: float = 6.5) -> bool:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    return calculate_entropy(image) > entropy_threshold


def process_images(folder_path: str):
    noisy_images = []
    all_images = [
        f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for filename in tqdm(all_images, desc="Analyzing image entropy"):
        path = os.path.join(folder_path, filename)
        try:
            if is_noisy(path):
                noisy_images.append(filename)
        except Exception as exc:
            print(f"Error analyzing {filename}: {exc}")

    return noisy_images


def main():
    folder = "pictures_with_noise"
    if not os.path.exists(folder):
        print(f"Directory '{folder}' does not exist.")
        return

    noisy = process_images(folder)
    print(f"Total noisy images detected: {len(noisy)}")
    with open("noisy_images.txt", "w", encoding="utf-8") as f:
        for img in noisy:
            f.write(f"{img}\n")


if __name__ == "__main__":
    main()
