from librairie import *


def focal_dice_loss(device, predictions, ground_truths, pixel_weight, gamma=2, alpha=1, num_classes=2, dims=(1, 2)):
    ground_truth_oh = F.one_hot(ground_truths, num_classes=num_classes)

    prediction_norm = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)

    # Intersection: |G ∩ P|. Shape: [B, num_classes]
    intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)

    # Summation: |G| + |P|. Shape: [B, num_classes].
    summation = (prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims))  # compute pixel by classe for GT+PREDICT

    # Dice Shape: [B, num_classes]
    dice = (2.0 * intersection) / (summation)

    # classwise
    # put at zero case where not class is present in predict and GT--> resultat nan
    dice = torch.nan_to_num(dice, nan=0.0)

    # number class by image
    num_class_present = torch.count_nonzero(summation, dim=1)

    # dice by image/class present
    # dice.sum(dim=1)--> sum dice of classes for each image
    # dice.sum(dim=1)/num_class_present--> mean Dice by classe
    # shape [batch_size,]
    dice = dice.sum(dim=1) / num_class_present

    # Compute the mean over the remaining axes (batch and classes).
    dice_mean = dice.mean()

    # prediction is logits, compute with log_softmax for to have log-probability (b,c,h,w)
    log_proba = F.log_softmax(predictions, dim=1)
    proba = torch.exp(log_proba)  # retriev probabilite --> (b,c,h,w)
    pt = proba.gather(dim=1, index=ground_truths.unsqueeze(1)).squeeze(
        1)  # proba--> shape(B,C,H,W) and ground_truths.unsqueeze(1)--> shape(B,C,H,W)
    alpha_classe = pixel_weight[ground_truths]
    focal = -alpha_classe * (1 - pt) ** gamma * torch.log(pt)
    focal_mean = focal.mean()
    return (1 - dice_mean) + focal_mean


def ce_dice_loss(predictions, ground_truths, num_classes=2, dims=(1, 2), smooth=1e-8):
    ground_truth_oh = F.one_hot(ground_truths, num_classes=num_classes)

    prediction_norm = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)

    # Intersection: |G ∩ P|. Shape: [B, num_classes]
    intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)

    # Summation: |G| + |P|. Shape: [B, num_classes].
    summation = ( prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims))  # compute pixel by classe for GT+PREDICT

    # Dice Shape: [B, num_classes]
    dice = (2.0 * intersection + smooth) / (summation + smooth)

    # Compute the mean over the remaining axes (batch and classes).
    dice_mean = dice.mean()

    # Compute cross-entropy loss.
    CE = F.cross_entropy(predictions, ground_truths)

    return (1.0 - dice_mean) + CE




