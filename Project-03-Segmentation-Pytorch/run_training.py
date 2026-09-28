from librairie import *
from Dataset.class_weight import find_class_weight
from Model.models import dic_model
from Entrainement.main import main
from Utils.utils import seed_everything, get_default_device
from Utils.utils import create_run_dir, selecte_model,create_dir_model


#create dir models
create_dir_model(TrainingConfig.LOG_DIR,TrainingConfig.Unet)
create_dir_model(TrainingConfig.LOG_DIR,TrainingConfig.Deeplab)
create_dir_model(TrainingConfig.LOG_DIR,TrainingConfig.DeeplabV3plus)


pixel_weight, classe_pixel_frequence = find_class_weight(device=DatasetConfig.DEVICE,path_train=DatasetConfig.DATA_TRAIN_LABELS,
                                                            number_class=DatasetConfig.NUM_CLASSES)

# For deterministic training
seed_everything(seed_value=42)

# Set default device to GPU if available.
DEVICE, GPU_AVAILABLE = get_default_device()

# select and Send model to device (GPU/CPU)
archi=TrainingConfig.DeeplabV3plus
backbone="resnet101" # put your backbone
model=selecte_model(dic_model,archi,backbone)


optimizer = Adam(model.parameters(),lr=TrainingConfig.LEARNING_RATE,weight_decay=1e-4)
scheduler=lr_scheduler.CosineAnnealingLR(optimizer,T_max=TrainingConfig.EPOCHS)


logdir=os.path.join(TrainingConfig.LOG_DIR,archi,backbone)
#os.makedirs(log_dir,exist_ok=True)
RUN_DIR,run_experience=create_run_dir(logdir)

model.to(DEVICE)

writer=SummaryWriter(log_dir=RUN_DIR)


print(f"logging : {RUN_DIR} - Experience:{run_experience}")

hparams = {
    "learning_rate": TrainingConfig.LEARNING_RATE,
    "batch_size": TrainingConfig.BATCH_SIZE,
    "weight_decay": 1e-4,
    "gamma": 2,
    "optimizer": optimizer.__class__.__name__,
    "scheduler": scheduler.__class__.__name__,
    "backbone": backbone,
    "methode_class_weight": f"inverse square root frequency)"}


metrics=main(RUN_DIR,
     writer,
     pixel_weight,
     model=model,
     optimizer=optimizer,
     device=DEVICE,
     pin_memory=GPU_AVAILABLE,
     scheduler=scheduler)

writer.add_hparams(hparams, metrics,run_name=run_experience)

writer.flush()
writer.close()