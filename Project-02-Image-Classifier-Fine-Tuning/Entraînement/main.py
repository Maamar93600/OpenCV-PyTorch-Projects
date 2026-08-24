
def main(DEVICE,MODEL,model,fine,log_dir,optimizer, summary_writer,  Config,scheduler=None):

    # Initialize data loader
    train_loader, valid_loader = get_data(
        batch_size=Config.batch_size,
        data_root=Config.data_root,
        num_workers=Config.num_workers)

    # Number of epochs to train.
    NUM_EPOCHS = Config.epochs_count
    print(f"USE DEVICE:{DEVICE}")

    best_accu =0.0
    patience=15
    counter=0

    # Epoch train & valid loss accumulator.
    epoch_train_loss = []
    epoch_valid_loss = []

    # Epoch train & valid accuracy accumulator.
    epoch_train_acc = []
    epoch_valid_acc = []

    # Trainig time measurement
    t_begin = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"Layer_2_LR update: {optimizer.param_groups[0]['lr']:.8f}|"
              f"Layer_3_LR update: {optimizer.param_groups[1]['lr']:.8f}|"
              f"Layer_4_LR update: {optimizer.param_groups[2]['lr']:.8f}|"
              f"FC_LR update: {optimizer.param_groups[3]['lr']:.8f}")
        train_loss, train_acc = train(DEVICE,Config, model, optimizer, train_loader, epoch + 1, NUM_EPOCHS)
        val_loss, val_accuracy = validate(DEVICE,Config, model, valid_loader, epoch + 1, NUM_EPOCHS)
        if scheduler is not None:
            scheduler.step()
            
 
        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)

        epoch_valid_loss.append(val_loss)
        epoch_valid_acc.append(val_accuracy)

        summary_writer.add_scalar("Loss/Train", train_loss, epoch)
        summary_writer.add_scalar("Accuracy/Train", train_acc, epoch)

        summary_writer.add_scalar("Loss/Validation", val_loss, epoch)
        summary_writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

        summary_writer.flush()
        

 
        if val_accuracy>best_accu:
            best_accu = val_accuracy
            print(f"\nModel Improved... Saving Model ... ", end="")
            print(f"Epoch: {epoch+1} | "
                  f"Val Acc: {best_accu:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(log_dir, f"{MODEL}_{fine}.pt"))
            print("Done.\n")
            counter=0
        else:
            counter +=1
    
        if counter>=patience:
            print(f"Early stopping à l'epoch {epoch+1}")
            print(f"Total time: {(time.time() - t_begin):.2f}s, Best Accu: {best_accu:.3f}")
            break

        print(f"{'='*72}\n")

    
    print(f"Total time: {(time.time() - t_begin):.2f}s, Best Accu: {best_accu:.3f}")

    return epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc