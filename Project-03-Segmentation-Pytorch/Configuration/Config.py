@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 12
    IMG_WIDTH: int = 384
    IMG_HEIGHT: int = 384

    DATA_TRAIN_IMAGES: str = os.path.join("data_project", "train", "images", "*.jpg")
    DATA_TRAIN_LABELS: str = os.path.join("data_project", "train", "masks", "*.png")
    DATA_VALID_IMAGES: str = os.path.join("data_project", "valid", "images", "*.jpg")
    DATA_VALID_LABELS: str = os.path.join("data_project", "valid", "masks", "*.png")
    DATA_TEST_IMAGES: str = os.path.join("data_project", "test", "images", "*.jpg")

    DEVICE = "cuda"


@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 8
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.0001
    MOMENTUM = 0.9
    CHECKPOINT_DIR: str = os.path.join('model_checkpoint')
    NUM_WORKERS: int = 4
    LOG_DIR = "log_P3_Seg"
    log_interval: int = 5
    test_interval: int = 1
    Unet = "Unet"
    Deeplab = "Deeplab"
    DeeplabV3plus = "DeepLabV3Plus"


@dataclass(frozen=True)
class InferenceConfig:
    BATCH_SIZE: int = 5
    NUM_BATCHES: int = 2



CLASS_COLORS = {
    "Background":   (0, 0, 0),
    "person":       (255, 0, 0),
    "bike":         (255, 128, 0),
    "car":          (255, 255, 0),
    "drone":        (255, 0, 255),
    "boat":         (0, 128, 255),
    "animal":       (128, 0, 255),
    "obstacle":     (128, 64, 0),
    "construction": (60, 60, 60),
    "vegetation":   (0, 200, 0),
    "road":         (0, 110, 110),
    "sky":          (0, 200, 255),
}


CLASS_id_COLORS = {
    idx: value
    for idx, value in enumerate(CLASS_COLORS.values())
}