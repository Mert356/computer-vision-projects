import shutil
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

import config


def download_images():
    archive = config.RAW_DIR / "images.tar.gz"
    if archive.exists():
        print(f"Already downloaded: {archive}")
        return archive

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.IMAGES_URL} (about 790 MB)")
    urllib.request.urlretrieve(config.IMAGES_URL, archive)
    return archive


def extract_images(archive):
    images_dir = config.RAW_DIR / "images"
    if images_dir.exists():
        print(f"Already extracted: {images_dir}")
        return images_dir

    print("Extracting images")
    with tarfile.open(archive) as tar:
        tar.extractall(config.RAW_DIR, filter='data')
    return images_dir


# Copies every image listed in the manifest to data/<split>/<label>/
def copy_to_splits(images_dir):
    df = pd.read_csv(config.MANIFEST_PATH)

    for path in df['path']:
        target = config.DATA_DIR / path
        target.parent.mkdir(parents=True, exist_ok=True)
        source = images_dir / Path(path).name
        shutil.copy(source, target)

    counts = df.groupby(['split', 'label']).size()
    print(f"Copied {len(df)} images")
    print(counts.to_string())


def main():
    archive = download_images()
    images_dir = extract_images(archive)
    copy_to_splits(images_dir)


if __name__ == '__main__':
    main()
