from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data"
MODELS_PATH = ROOT / "models"
RESULTS_PATH = ROOT / "results"
LOGS_PATH = ROOT / "logs"

IMAGE_SIZE = 224
EMBEDDING_SIZE = 512

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5

RANDOM_STATE = 42