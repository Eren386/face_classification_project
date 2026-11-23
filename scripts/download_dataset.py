import json
import os
import opendatasets as od


def main():
    dataset_url = "https://www.kaggle.com/datasets/mohannadaymansalah/stable-diffusion-dataaaaaaaaa"

    print("=====================================================")
    print(" KAGGLE STABLE DIFFUSION DATASET DOWNLOADER")
    print("=====================================================")

    username = input("Kaggle Username: ").strip()
    key = input("Kaggle Key (API Token): ").strip()

    with open("kaggle.json", "w", encoding="utf-8") as f:
        json.dump({"username": username, "key": key}, f)

    print("\nStarting download...")

    try:
        od.download(dataset_url, data_dir="./")
        print("\nDownload and extraction completed successfully!")
        print("Run scripts/process_new_fakes.py to crop faces into Data_cropped/fake_faces.")
    except Exception as exc:
        print(f"\n[!] Download failed: {exc}")
    finally:
        if os.path.exists("kaggle.json"):
            os.remove("kaggle.json")


if __name__ == "__main__":
    main()
