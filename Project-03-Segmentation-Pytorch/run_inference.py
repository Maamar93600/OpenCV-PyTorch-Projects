from librairie import *
from Model.models import dic_model
from Utils.utils import get_default_device
from Evaluation.selecter_model_inference import selecte_model_inference
from Dataset import get_dataloader
from Configuration.Config import CLASS_id_COLORS,InferenceConfig,TrainingConfig
from Evaluation import display_image_and_mask_inferences



DEVICE, GPU_AVAILABLE = get_default_device()

archi="DeepLabV3Plus"
backbone="resnet101"

load_model_inférence=selecte_model_inference(dic_model,archi,backbone,"Run_0",DEVICE)


_, valid_loader = get_dataloader(batch_size=InferenceConfig.BATCH_SIZE,num_workers=TrainingConfig.NUM_WORKERS,pin_memory=GPU_AVAILABLE)


with torch.inference_mode():
    batch_img, batch_mask = next(iter(valid_loader))
    pred_all = load_model_inférence(batch_img.to(DEVICE))
    pred_masks = pred_all.argmax(dim=1).cpu()

display_image_and_mask_inferences(batch_img.clone(), batch_mask, pred_masks, ROW=5, COLS=4, color_map=CLASS_id_COLORS)




