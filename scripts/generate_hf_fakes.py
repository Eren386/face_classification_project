import io
import os
import random
import time
import requests
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

ANGLES = [
    "A selfie of",
    "A close-up portrait of",
    "A CCTV security camera shot of",
    "A wide-angle shot of",
    "A low-angle dramatic shot of",
    "A webcam snapshot of",
]
DEMOGRAPHICS = [
    "a 20-year-old student",
    "a middle-aged professional",
    "an elderly person with wrinkles",
    "a teenager",
    "a fashion model",
    "an ordinary person on the street",
    "a tired worker",
    "a person with freckles and messy hair",
]
LIGHTING = [
    "under harsh neon lights",
    "with a camera flash reflecting on the skin",
    "in natural cinematic sunlight",
    "in a dark, moody room",
    "with fluorescent office lighting",
    "at golden hour",
]
STYLES = [
    "highly detailed, photorealistic, 8k resolution, raw photo",
    "amateur photography, slightly blurred, noisy, taken with an old phone",
    "polaroid style, vintage colors",
    "ultra-realistic, sharp focus, skin pores visible",
]


def generate_prompt() -> str:
    return (
        f"{random.choice(ANGLES)} {random.choice(DEMOGRAPHICS)} "
        f"{random.choice(LIGHTING)}, {random.choice(STYLES)}."
    )


def query_api(payload: dict, headers: dict):
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        print(f"   [!] API Error {response.status_code}: {response.text}")
        return None
    return response.content


def main():
    print("=====================================================")
    print(" SYNTHETIC FACE GENERATOR (FLUX.1-schnell via HF API)")
    print("=====================================================")

    hf_token = input("Enter your HuggingFace Access Token (hf_...): ").strip()
    headers = {"Authorization": f"Bearer {hf_token}"}

    target_dir = os.path.join("Data_cropped", "hf_flux_fakes")
    os.makedirs(target_dir, exist_ok=True)

    num_images = 200
    print(f"\nGenerating {num_images} synthetic portraits...\n")

    success_count = 0
    for i in range(num_images):
        prompt = generate_prompt()
        print(f"[{i + 1}/{num_images}] Prompt: {prompt}")

        image_bytes = query_api({"inputs": prompt}, headers)
        if image_bytes:
            try:
                image = Image.open(io.BytesIO(image_bytes))
                save_path = os.path.join(target_dir, f"flux_fake_{success_count}.jpg")
                image.save(save_path)
                success_count += 1
                print(f"   Saved: {save_path}")
            except Exception as exc:
                print(f"   [!] Failed to decode image: {exc}")

        # Rate-limit delay for free HuggingFace inference endpoints
        time.sleep(3)

    print(f"\nCompleted! Generated {success_count} FLUX portraits in '{target_dir}'.")


if __name__ == "__main__":
    main()
