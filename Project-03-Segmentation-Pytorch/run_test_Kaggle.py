from librairie import *
from Model.models import dic_model
from Utils.utils import get_default_device
from Evaluation import selecter_model_inferences
from Configuration.Config import CLASS_id_COLORS,InferenceConfig,TrainingConfig,DatasetConfig
from Dataset import CustomSegDatasets
from Test_Kaggle import display_image_test_and_pred_mask_inferences
from Test_Kaggle.rle import rle_encode



DEVICE, GPU_AVAILABLE = get_default_device()

# Load best model Dice
archi = "DeepLabV3Plus"
backbone = "resnet101"

load_model_inférence = selecter_model_inference(dic_model, archi, backbone, "Run_0", DEVICE)

# Batch Test load
IMAGE_SIZE = (DatasetConfig.IMG_WIDTH,DatasetConfig.IMG_HEIGHT)
#Liste des images test
path_data_test=glob(DatasetConfig.DATA_TEST_IMAGES)

dataset_test = CustomSegDatasets(image_paths=path_data_test, mask_paths=None,image_size=IMAGE_SIZE, is_train=False)
test_loader = DataLoader(dataset_test, batch_size=InferenceConfig.BATCH_SIZE, pin_memory=GPU_AVAILABLE,
                         num_workers=TrainingConfig.NUM_WORKERS, shuffle=False)

# Mode eval
load_model_inférence.eval()
# move Model to device
load_model_inférence.to(DEVICE)
PREDICTION = []

with torch.no_grad():
    for batch_input in test_loader:
        # move data same device to model

        pred_mask = load_model_inférence(batch_input.to(DEVICE))

        pred_mask_id = pred_mask.argmax(dim=1)

        pred_mask_id = pred_mask_id.cpu()

        PREDICTION.extend(pred_mask_id.numpy())


d = {}

for idx, path_img in enumerate(path_data_test):
    img_read = cv2.imread(path_img, cv2.IMREAD_COLOR)
    H, W = img_read.shape[0:2]
    PREDICTION[idx] = cv2.resize(PREDICTION[idx], dsize=(W, H), interpolation=cv2.INTER_NEAREST)
    name = os.path.basename(path_img).split(".")[0]

    for j in range(12):
        N = f"{name}_{j}"
        d[N] = rle_encode(PREDICTION[idx] == j)

DATAFRAME = pd.DataFrame(data=d.items(), columns=["ImageID", "EncodedPixels"])
DATAFRAME.to_csv("Maamarv5_Sumission.csv", index=False)




LISTE_IMAGE_TEST=path_data_test
display_image_test_and_pred_mask_inference(LISTE_IMAGE_TEST,PREDICTION,color_map=CLASS_id_COLORS)

