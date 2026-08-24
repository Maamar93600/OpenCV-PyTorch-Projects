@dataclass
class Configuration:
    """
    Describes the configuration of the training process.
    """

    # =========================
    # Dataset
    # =========================
    batch_size: int = 32
    data_root: str = "data"
    NUM_CLASSES: int = 13
    IMG_HEIGHT: int = 224
    IMG_WIDTH: int = 224

    # =========================
    # Training
    # =========================
    epochs_count: int = 100
    num_workers: int = 0
    device: str = "cuda"

    # =========================
    # Learning rates
    # =========================
    LR_layer: float = 1e-5
    LR_FC: float = 5e-4
    MOMENTUM: float = 0.9

    # =========================
    # Logging / evaluation
    # =========================
    log_interval: int = 5
    test_interval: int = 1
    LOG_DIR: str = "LOGS_PROJET_2_Classification"

    # =========================
    # Fine-tuning
    # =========================
    fine_tune_start: int = 3
    fine_tune_start_Effi: int = 3

    # =========================
    # Models
    # =========================
    model_names = ["resnet50"]
    model_name_effi = ["efficientnet_b0"]

    # =========================
    # Class weights
    # =========================
    Poids = KenyanFood13Dataset(data_root,"train.csv").poids