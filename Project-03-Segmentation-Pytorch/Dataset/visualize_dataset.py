from librairie import *

def display_image_and_mask_bis(valid_loader, ROW=3, COLS=3, color_map=CLASS_COLORS):
    batch_images, batch_masks = next(iter(valid_loader))
    batch_images = denormalize(batch_images).permute(0, 2, 3, 1).numpy()
    batch_masks = batch_masks.numpy()

    ROW = min(ROW, len(batch_images))
    title = ['GT Image', 'GT Mask', 'Overlayed Mask']
    fig, ax = plt.subplots(ROW, 3, figsize=(15, ROW * 4))
    for i in range(ROW):
        if i == 0:
            ax[i, 0].set_title(title[0], fontsize=14)
            ax[i, 1].set_title(title[1], fontsize=14)
            ax[i, 2].set_title(title[2], fontsize=14)

        image = batch_images[i]
        grayscale_gt_mask = batch_masks[i]
        rgb_gt_mask = num_to_rgb(grayscale_gt_mask, color_map=color_map)
        overlayed_image = image_overlay(image, rgb_gt_mask)

        ax[i, 0].imshow(image)
        ax[i, 0].axis("off")
        ax[i, 1].imshow(rgb_gt_mask)
        ax[i, 1].axis("off")
        ax[i, 2].imshow(overlayed_image)
        ax[i, 2].axis("off")
    plt.tight_layout()


def num_to_rgb(num_arr, color_map=CLASS_COLORS):
    single_layer = np.squeeze(num_arr)
    # other methode : H, W = img.shape[:2]  et np.zeros((H, W, 3))
    output = np.zeros(num_arr.shape[:2] + (3,))  # on addition tuple pour ajouter 3iem dim

    for k in color_map.keys():
        output[single_layer == k] = color_map[k]

    return np.float32(output) / 255.0  # return a floating point array in range [0.0, 1.0]


# Function to overlay a segmentation map on top of an RGB image.
def image_overlay(image, segmented_image):

    alpha = 1 # Transparency for the original image.
    beta  = 0.7 # Transparency for the segmentation map.
    gamma = 0.0 # Scalar added to each sum.

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.clip(image, 0.0, 1.0)


def denormalize(tensors, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalization parameters for pre-trained PyTorch models
    Denormalizes image tensors using mean and std"""

    for c in range(3):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors, min=0.0, max=1.0)




