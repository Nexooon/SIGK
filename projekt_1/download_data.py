import os
import urllib.request
import zipfile

URLS = {
    "valid": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    # Opcjonalne, można zakomentować jeśli nie trenujemy:
    "train": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
}


def download_and_extract(name, url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, f"DIV2K_{name}_HR.zip")

    if not os.path.exists(zip_path):
        print(f"Pobieranie zbioru '{name}' z {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Pobrano pomyślnie!")

    print(f"Rozpakowywanie {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"Gotowe, zdjęcia znajdują się w katalogu: {target_dir}\n")


if __name__ == "__main__":
    download_and_extract("valid", URLS["valid"], "data/valid")
    if "train" in URLS:
        download_and_extract("train", URLS["train"], "data/train")
