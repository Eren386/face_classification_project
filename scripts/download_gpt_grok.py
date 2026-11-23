import os
import shutil
import subprocess
import sys
import zipfile

# High-quality AI-generated faces dataset on Kaggle
DATASET_NAME = "shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset"


def main():
    print("=====================================================")
    print(" KAGGLE SMART DOWNLOADER (DISK-SAFE 10,000 CAP)")
    print("=====================================================")

    os.environ["KAGGLE_USERNAME"] = input("Kaggle Username: ").strip()
    os.environ["KAGGLE_KEY"] = input("Kaggle Key (API Token): ").strip()

    print("\nDownloading dataset archive from Kaggle...")

    try:
        scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
        kaggle_exe = os.path.join(scripts_dir, "kaggle.exe")
        if not os.path.exists(kaggle_exe):
            kaggle_exe = "kaggle"
        subprocess.run([kaggle_exe, "datasets", "download", "-d", DATASET_NAME], check=True)
    except Exception as exc:
        print(f"\n[!] Download error. Check Kaggle credentials/CLI installation: {exc}")
        return

    zip_file = DATASET_NAME.split("/")[-1] + ".zip"
    if not os.path.exists(zip_file):
        print("[!] Downloaded zip file not found.")
        return

    print("\nDownload complete. Streaming up to 10,000 AI-generated images from archive...")

    extract_count = 0
    max_extract = 10000
    target_dir = os.path.join("Data_cropped", "gpt_midjourney_fakes")
    os.makedirs(target_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file, "r") as zf:
        all_files = zf.namelist()
        fake_files = [
            f for f in all_files if ("fake" in f.lower() or "ai" in f.lower() or "synth" in f.lower())
        ]
        if not fake_files:
            fake_files = all_files

        for file in fake_files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                file_name = os.path.basename(file)
                if not file_name:
                    continue

                with zf.open(file) as src, open(
                    os.path.join(target_dir, f"new_fake_{extract_count}.jpg"), "wb"
                ) as dst:
                    shutil.copyfileobj(src, dst)

                extract_count += 1
                if extract_count % 1000 == 0:
                    print(f"   Extracted: {extract_count} / {max_extract}")

                if extract_count >= max_extract:
                    break

    print(f"\nSuccessfully extracted {extract_count} synthetic images into '{target_dir}'.")
    print("Removing temporary zip archive to save disk space...")
    os.remove(zip_file)
    print("Done.")


if __name__ == "__main__":
    main()
