from librairie import *


def TP_FP_FN(predictions, ground_truths, num_classes=2, dims=(1, 2)):
    PRED_one = F.one_hot(predictions, num_classes=num_classes)  # --> shape (B,H,W,C)
    GT_one = F.one_hot(ground_truths, num_classes=num_classes)  # --> shape (B,H,W,C)

    TP = (PRED_one * GT_one).sum(dim=dims)  # its intersection
    # compute FP and FN
    FP = (PRED_one & ~GT_one).sum(dim=dims)  # --> shape (B,C)
    FN = (~PRED_one & GT_one).sum(dim=dims)  # --> shape (B,C)

    return TP, FP, FN


# function mean_IoU for Validation
def mean_iou(predictions, ground_truths, num_classes=2, dims=(1, 2)):
    GT = F.one_hot(ground_truths, num_classes=num_classes)

    # shape (B,H,W)--> (B,H,W,C)
    PRED = F.one_hot(predictions, num_classes=num_classes)

    intersection = (PRED * GT).sum(dim=dims)

    summation = (PRED.sum(dim=dims) + GT.sum(dim=dims))

    union = summation - intersection

    IoU = intersection / union

    # classwise
    # put at zero case where not class is present in predict and GT--> resultat nan
    IoU = torch.nan_to_num(IoU, nan=1.0)

    # number class by image
    # num_class_present=torch.count_nonzero(summation,dim=1)# number of class by image
    # num_class=IoU.shape[1]

    # IoU by image/class present
    # IoU.sum(dim=1)--> sum IoU of classes for each image
    # IoU.sum(dim=1)/num_class_present--> mean IoU by classe
    # shape [batch_size,]
    # IoU_M=IoU.sum(dim=1)/num_class

    # Compute the mean over the remaining axes (batch and classes).
    # IoU_mean = IoU_M.mean()
    IoU_mean = IoU.mean()

    return IoU_mean


# function Dice_coef for Train
def Dice_coef(predictions, ground_truths, num_classes=2, dims=(1, 2)):
    GT = F.one_hot(ground_truths, num_classes=num_classes)

    # shape (B,H,W)--> (B,H,W,C)
    PRED = F.one_hot(predictions, num_classes=num_classes)

    intersection = (PRED * GT).sum(dim=dims)

    summation = (PRED.sum(dim=dims) + GT.sum(dim=dims))

    Dice = (2 * intersection) / summation

    # classwise
    # put at zero case where not class is present in predict and GT--> resultat nan
    Dice = torch.nan_to_num(Dice, nan=1.0)

    # number class by image
    # num_class_present=torch.count_nonzero(summation,dim=1)
    # num_class=Dice.shape[1]

    # IoU by image/class present
    # IoU.sum(dim=1)--> sum IoU of classes for each image
    # IoU.sum(dim=1)/num_class_present--> mean IoU by classe
    # shape [batch_size,]
    # Dice=Dice.sum(dim=1)/num_class

    # Compute the mean over the remaining axes (batch and classes).
    Dice_mean = Dice.mean()

    return Dice_mean





