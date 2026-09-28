from librairie import *
from Utils import num_to_rgb,image_overlay
from Configuration.Config import CLASS_id_COLORS


def display_image_test_and_pred_mask_inferences(list_images_test, pred_masks, ROW=5, COLS=3, color_map=CLASS_id_COLORS):
    LISTE_RANDOME = random.sample(range(len(pred_masks)), ROW)

    title = ['GT Image', 'PREDICT Mask', 'Overlayed Mask']
    fig, ax = plt.subplots(ROW, COLS, figsize=(20, ROW * 4))
    for i in range(ROW):
        if i == 0:
            ax[i, 0].set_title(title[0], fontsize=14)
            ax[i, 1].set_title(title[1], fontsize=14)
            ax[i, 2].set_title(title[2], fontsize=14)

        image_test_read = cv2.imread(list_images_test[LISTE_RANDOME[i]], cv2.IMREAD_COLOR)[:,:,::-1].astype('float32') / 255.0
        mask_predict_to_rgb = num_to_rgb(pred_masks[LISTE_RANDOME[i]], color_map=color_map)
        overlayed_image = image_overlay(image_test_read, mask_predict_to_rgb)

        ax[i, 0].imshow(np.clip(image_test_read, 0.0, 1.0))
        ax[i, 0].axis("off")
        ax[i, 1].imshow(mask_predict_to_rgb)
        ax[i, 1].axis("off")
        ax[i, 2].imshow(overlayed_image)
        ax[i, 2].axis("off")
    plt.tight_layout()


