import os
import cv2
import matplotlib.pyplot as plt


def load_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    real_path = os.path.join("Data", "real_faces")
    fake_path = os.path.join("Data", "fake_faces")

    real_files = os.listdir(real_path) if os.path.exists(real_path) else []
    fake_files = os.listdir(fake_path) if os.path.exists(fake_path) else []

    if not real_files or not fake_files:
        raise RuntimeError("Either 'Data/real_faces' or 'Data/fake_faces' is empty or missing.")

    real_image = load_image(os.path.join(real_path, real_files[0]))
    fake_image = load_image(os.path.join(fake_path, fake_files[0]))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(real_image)
    plt.title("Real Face")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(fake_image)
    plt.title("Fake (AI-Generated) Face")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
