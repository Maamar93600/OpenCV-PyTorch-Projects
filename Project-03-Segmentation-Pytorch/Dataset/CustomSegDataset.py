from librairie import *

# Custom Class for creating training and validation (segmentation) dataset objects.

class CustomSegDatasets(Dataset):

    def __init__(self, *, image_size, image_paths, mask_paths=None, is_train=False):

        self.image_size = image_size
        self.image_paths = image_paths
        self.mask_paths = mask_paths  # None in case of test set where only images are available.
        self.is_train = is_train
        if self.mask_paths:
            self.Mode_canal = np.unique([Image.open(i).mode for i in self.mask_paths])

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        Transformation = []
        if self.is_train:
            Transformation.extend([A.HorizontalFlip(p=0.5),
                                   A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.5),
                                   A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                                   A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                                   ])

        Transformation.extend([A.Normalize(mean=mean, std=std),
                               A.ToTensorV2()  # convert HWC-->CHW
                               ])
        self.transforms = A.Compose(Transformation)

    def __len__(self):
        return len(self.image_paths)

    def load_file(self, file_path, flags=cv2.IMREAD_COLOR, interpolation=cv2.INTER_NEAREST):
        if flags == cv2.IMREAD_GRAYSCALE:
            file = cv2.imread(file_path, flags)
        else:
            file = cv2.imread(file_path, flags)[:, :, ::-1]
        file = cv2.resize(file, self.image_size, interpolation=interpolation)
        return file

    def __getitem__(self, index):

        # Get image and mask path.
        image_path = self.image_paths[index]

        # Load image using opencv and convert image format from BGR to RGB and resize.
        image = self.load_file(image_path, flags=cv2.IMREAD_COLOR, interpolation=cv2.INTER_CUBIC)

        if self.mask_paths is not None:  # True for Training and validation set.
            # Get mask path.
            mask_path = self.mask_paths[index]

            if self.Mode_canal == "RGB":
                mask = self.load_file(mask_path, flags=cv2.IMREAD_COLOR, interpolation=cv2.INTER_NEAREST)
                mask = rgb_to_grayscale(mask)
            else:
                mask = self.load_file(mask_path, flags=cv2.IMREAD_GRAYSCALE, interpolation=cv2.INTER_NEAREST)

            # Apply Preprocessing (+ Augmentations) transformations to image-mask pair
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask'].to(torch.long)
            return image, mask

        else:  # For Test set.

            # Apply Preprocessing transformations to image.
            transformed = self.transforms(image=image)

            image = transformed['image']

            return image

