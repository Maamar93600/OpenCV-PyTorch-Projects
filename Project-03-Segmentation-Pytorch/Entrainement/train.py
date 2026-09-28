from librairie import *

def training(
        pixel_weight,
        model,
        loader,
        optimizer,
        scaler,
        num_classes,
        device="cpu",
        epoch_idx=1,
        total_epochs=100,
):
    # Change model mode.
    model.train()

    loss_record = MeanMetric()
    metric_record = MeanMetric()

    loader_len = len(loader)

    # with tqdm(total=loader_len, ncols=122) as tq:
    #   tq.set_description(f"Train :: Epoch: {epoch_idx}/{total_epochs}")
    status = f"Train:\t{bold}Epoch: {epoch_idx}/{total_epochs}{reset}"
    pbar = tqdm(loader, bar_format='{l_bar}{bar:10}{r_bar}')
    pbar.set_description(status)

    for data, target in pbar:
        # tq.update(1)

        # Send data and target to GPU device if available.
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # Reset parameters gradient to zero.
        optimizer.zero_grad()

        with amp.autocast(device_type=device.type):  # Autocasting for mixed-precision training.
            # Perform Forward pass through the model.
            logits = model(data)

            # Calculate Combo loss (Segmentation specific loss (Dice) + cross entropy)
            loss = focal_dice_loss(device, logits, target, pixel_weight, num_classes=num_classes)

        # Calculate gradients w.r.t training parameters on scaled loss.
        scaler.scale(loss).backward()

        # Update parameters using gradients
        scaler.step(optimizer)

        # Updates the scale for the next iteration.
        scaler.update()

        # Detach the output "logits" tensor from the graph.
        logits = logits.detach()

        # Get the index across channel axis of the max logit score.
        # We can directly call argmax because: Softmax(logit).argmax() == logit.argmax()
        # shape-->(B,H,W)
        pred_idx = logits.argmax(dim=1)

        # Calculate Segmentation specific metric Dice mean
        metric = Dice_coef(pred_idx, target, num_classes=num_classes)

        # Record loss and IoU metric.
        loss_record.update(loss.detach().cpu(), weight=data.shape[0])
        metric_record.update(metric.cpu(), weight=data.shape[0])

        # Update progress bar description.
        pbar.set_postfix_str(s=f"Train Loss: {loss_record.compute():.4f}, Train Dice: {metric_record.compute():.4f}")

    # Get mean Focalloss+Dice, and Dice score.
    epoch_loss = loss_record.compute()
    epoch_metric = metric_record.compute()

    return epoch_loss, epoch_metric



