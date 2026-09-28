from librairie import *

def get_dataloader(batch_size=4, num_workers=0, pin_memory=False):

    NUM_CLASSES = DatasetConfig.NUM_CLASSES

    # OpenCV .resize(..) method accepts new size in format (new_width, new_height).
    IMAGE_SIZE = (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT)

    # Training image and mask paths.
    train_images = sorted(glob(f"{DatasetConfig.DATA_TRAIN_IMAGES}"))
    train_masks  = sorted(glob(f"{DatasetConfig.DATA_TRAIN_LABELS}"))

    # Validation image and mask paths.
    valid_images = sorted(glob(f"{DatasetConfig.DATA_VALID_IMAGES}"))
    valid_masks  = sorted(glob(f"{DatasetConfig.DATA_VALID_LABELS}"))

    # Create training dataset and dataloader.
    train_dataset = CustomSegDataset(image_paths=train_images, mask_paths=train_masks, is_train=True,image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,  pin_memory=pin_memory,num_workers=num_workers, drop_last=True, shuffle=True)

    # Create validation dataset and dataloader.
    valid_dataset = CustomSegDataset(image_paths=valid_images, mask_paths=valid_masks, is_train=False,image_size=IMAGE_SIZE)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,  pin_memory=pin_memory,num_workers=num_workers, shuffle=False)

    return train_loader, valid_loader
