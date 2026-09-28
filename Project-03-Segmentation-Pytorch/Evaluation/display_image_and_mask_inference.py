from librairie import *

def display_image_and_mask_inferences(batch_images, batch_masks, pred_masks, ROW=5, COLS=4, color_map=CLASS_id_COLORS):
    batch_images_dnor = denormalize(batch_images).permute(0, 2, 3, 1).numpy()
    batch_masks = batch_masks.numpy()
    print(ROW)

    title = ['GT Image', 'GT Mask', 'PREDICT Mask', 'Overlayed Mask']
    fig, ax = plt.subplots(ROW, COLS, figsize=(20, ROW * 5))
    for i in range(ROW):
        if i == 0:
            ax[i, 0].set_title(title[0], fontsize=14)
            ax[i, 1].set_title(title[1], fontsize=14)
            ax[i, 2].set_title(title[2], fontsize=14)
            ax[i, 3].set_title(title[3], fontsize=14)

        mask_gt_to_rgb = num_to_rgb(batch_masks[i], color_map=color_map)
        mask_predict_to_rgb = num_to_rgb(pred_masks[i], color_map=color_map)
        overlayed_image = image_overlay(batch_images_dnor[i], mask_predict_to_rgb)

        ax[i, 0].imshow(batch_images_dnor[i])
        ax[i, 0].axis("off")
        ax[i, 1].imshow(mask_gt_to_rgb)
        ax[i, 1].axis("off")
        ax[i, 2].imshow(mask_predict_to_rgb)
        ax[i, 2].axis("off")
        ax[i, 3].imshow(overlayed_image)
        ax[i, 3].axis("off")
    plt.tight_layout()


