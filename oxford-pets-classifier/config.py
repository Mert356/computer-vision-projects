from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MANIFEST_PATH = DATA_DIR / "split_manifest.csv"
RAW_DIR = DATA_DIR / "raw"
IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
OUTPUT_DIR = PROJECT_DIR / "outputs"
RESULTS_DIR = PROJECT_DIR / "results"
CHECKPOINT_PATH = OUTPUT_DIR / "best.pt"

# One cat and one dog from the test split shown in results/samples.png
SAMPLE_IMAGES = ("test/cat/Bengal_1.jpg", "test/dog/samoyed_1.jpg")

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
