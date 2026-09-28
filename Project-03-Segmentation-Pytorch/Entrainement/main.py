from librairie import *
from Entrainement.train import training
from Entrainement.validate import validate

def main(log_dir, summary_writer, pixel_weight, model, optimizer, device, pin_memory, scheduler=None):
    # Create Dataloader.
    train_loader, valid_loader = get_dataloader(batch_size=TrainingConfig.BATCH_SIZE, pin_memory=pin_memory,
                                                num_workers=TrainingConfig.NUM_WORKERS)

    # Save the model if best_Dice improves.
    Best_Dice = 0.0
    patience = 15
    counter = 0

    # Epoch train & valid loss accumulator.
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_mean_iou = []
    total_RECALL_CLASS = torch.zeros(DatasetConfig.NUM_CLASSES, device=device)
    total_PRECISION_CLASS = torch.zeros(DatasetConfig.NUM_CLASSES, device=device)
    total_IoU_CLASS = torch.zeros(DatasetConfig.NUM_CLASSES, device=device)

    total_epochs = TrainingConfig.EPOCHS

    scaler = amp.GradScaler()

    # Training Loop.
    for epoch in range(total_epochs):
        epochs_done = epoch + 1

        # Memory Cleanup.
        torch.cuda.empty_cache()
        gc.collect()

        # Train one epoch.
        train_loss, train_metric = training(pixel_weight, model=model, loader=train_loader, optimizer=optimizer,
                                            scaler=scaler,
                                            num_classes=DatasetConfig.NUM_CLASSES, device=device, epoch_idx=epoch + 1,
                                            total_epochs=total_epochs)

        # Peform model validation.
        valid_loss, valid_metric, valid_epoch_metric_Dice, Recall_Class, Precision_Classe, IoU_Classe, DICE = validate(
            pixel_weight, model=model, loader=valid_loader, device=device,
            num_classes=DatasetConfig.NUM_CLASSES, epoch_idx=epoch + 1, total_epochs=total_epochs)

        if scheduler is not None:
            scheduler.step()

        # Epoch train & valid loss accumulator.
        epoch_train_loss.append(train_loss)
        epoch_valid_loss.append(valid_loss)
        epoch_mean_iou.append(valid_metric)

        total_RECALL_CLASS += Recall_Class
        total_PRECISION_CLASS += Precision_Classe
        total_IoU_CLASS += IoU_Classe

        summary_writer.add_scalar("Loss/Train", train_loss, epoch)
        summary_writer.add_scalar("Loss/Validation", valid_loss, epoch)
        summary_writer.add_scalar("Valid_Mean_IoU", valid_metric, epoch)
        for Class, i in zip(CLASS_COLORS, IoU_Classe):
            summary_writer.add_scalar(f"Class_{Class}/IoU", i, epoch)

        for Class, i in zip(CLASS_COLORS, Recall_Class):
            summary_writer.add_scalar(f"Class_{Class}/Recall", i, epoch)

        for Class, i in zip(CLASS_COLORS, Precision_Classe):
            summary_writer.add_scalar(f"Class_{Class}/Precision", i, epoch)

        print(DICE)

        # Create model and optimizer checkpoint.
        if valid_epoch_metric_Dice > Best_Dice:
            Best_Dice = valid_epoch_metric_Dice
            Best_IoU = valid_metric
            print("Model Improved. Saving...", end="")
            print(f"Epoch: {epoch + 1} | "
                  f"Valid Best Dice: {Best_Dice:.4f} | "
                  f"Valid Best Mean_IoU: {Best_IoU:.4f}")

            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "model": model.state_dict(),
                "scaler": scaler.state_dict(),
            }
            torch.save(checkpoint_dict, os.path.join(log_dir, "ckpt.tar"))
            del checkpoint_dict
            print("Done.\n")
            counter = 0

        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping à l'epoch {epoch + 1}")

            break

        print(f"{'=' * 72}\n")

    epoch_mean_iou = sum(epoch_mean_iou) / epochs_done

    metrics = {"best_Dice": Best_Dice,
               "best_mIoU": Best_IoU,
               "Mean IoU Validation": epoch_mean_iou
               }
    print(f"Epoch: {epoch + 1} | "
          f"Valid Best Dice: {Best_Dice:.4f} | "
          f"Valid Best Mean_IoU: {Best_IoU:.4f}")

    return metrics

