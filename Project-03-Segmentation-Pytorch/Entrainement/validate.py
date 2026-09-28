from librairie import *

def validate(
        pixel_weight,
        model,
        loader,
        device,
        num_classes,
        epoch_idx,
        total_epochs,
):
    # Change model mode.
    model.eval()

    loss_record = MeanMetric()
    metric_record = MeanMetric()
    metric_Dice_record = MeanMetric()

    FN_CLASSE = torch.zeros(num_classes, device=device)
    FP_CLASSE = torch.zeros(num_classes, device=device)
    TP_CLASSE = torch.zeros(num_classes, device=device)

    loader_len = len(loader)

    # with tqdm(total=loader_len, ncols=122) as tq:
    #    tq.set_description(f"Valid :: Epoch: {epoch_idx}/{total_epochs}")
    status = f"Valid:\t{bold}Epoch: {epoch_idx}/{total_epochs}{reset}"
    pbar = tqdm(loader, bar_format='{l_bar}{bar:10}{r_bar}')
    pbar.set_description(status)

    for data, target in pbar:
        # tq.update(1)

        # Send data and target to GPU device if available.
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        with torch.no_grad():
            # Perform Forward pass through the model. Output is a dictionary.
            logits = model(data)

        # Calculate Combo loss (Segmentation specific loss (Dice) + cross entropy)
        loss = focal_dice_loss(device, logits, target, pixel_weight, num_classes=DatasetConfig.NUM_CLASSES)

        # Get the index across channel axis of the max logit score.
        # We can directly call argmax because: Softmax(logit).argmax() == logit.argmax()
        pred_idx = logits.argmax(dim=1)

        # Calculate Segmentation specific metric (Dice and IoU).
        metric_Dice = Dice_coef(pred_idx, target, num_classes=num_classes)
        metric_mean_IoU = mean_iou(pred_idx, target, num_classes=num_classes)

        TP, FP, FN = TP_FP_FN(pred_idx, target, num_classes=num_classes, dims=(1, 2))

        FN_CLASSE += FN.sum(dim=0)
        FP_CLASSE += FP.sum(dim=0)
        TP_CLASSE += TP.sum(dim=0)

        # Record loss and IoU metric.
        loss_record.update(loss.cpu(), weight=data.shape[0])
        metric_record.update(metric_mean_IoU.cpu(), weight=data.shape[0])
        metric_Dice_record.update(metric_Dice.cpu(), weight=data.shape[0])

        # Update progress bar description to display epoch log.
        pbar.set_postfix_str(
            s=f"Valid Loss: {loss_record.compute():.4f}, Valid DICE: {metric_Dice_record.compute():.4f} ,Valid Mean_IoU: {metric_record.compute():.4f}")

    Recall_Class = TP_CLASSE / (TP_CLASSE + FN_CLASSE)
    Precision_Classe = TP_CLASSE / (TP_CLASSE + FP_CLASSE)
    IoU_Classe = TP_CLASSE / (TP_CLASSE + FP_CLASSE + FN_CLASSE)
    DICE = 2 * TP_CLASSE / (2 * TP_CLASSE + FP_CLASSE + FN_CLASSE)
    # Compute Epoch loss, Dice score.
    valid_epoch_loss = loss_record.compute()
    valid_epoch_metric = metric_record.compute()
    valid_epoch_metric_Dice = metric_Dice_record.compute()

    return valid_epoch_loss, valid_epoch_metric, valid_epoch_metric_Dice, Recall_Class, Precision_Classe, IoU_Classe, DICE.mean()



