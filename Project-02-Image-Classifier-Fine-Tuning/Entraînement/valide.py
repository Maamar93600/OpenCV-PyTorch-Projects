def validate(DEVICE,Config,model,test_loader,epoch_idx,total_epochs):
    
    model.eval()
    model=model.to(DEVICE)
    
    acc_metric = MulticlassAccuracy(num_classes=Config.NUM_CLASSES, average="micro")
    mean_metric = MeanMetric()
    
    status = f"Valid:\t{bold}Epoch: {epoch_idx}/{total_epochs}{reset}"
    
    prog_bar = tqdm(test_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    prog_bar.set_description(status)
    
    for data, target in prog_bar:
    
        # Send data and target to appropriate device.
        data, target = data.to(DEVICE), target.to(DEVICE)
    
        # Get the model's predicted logits.
        with torch.no_grad():
            output = model(data)
    
        # Compute the CE-Loss.
        test_loss = F.cross_entropy(output, target).item()
    
        # Convert model's logits to probability scores.
        prob = F.softmax(output, dim=1)
    
        # Get the class id for the maximum score.
        pred_idx = prob.detach().argmax(dim=1)
    
        # Batch validation loss.
        batch_loss = mean_metric(test_loss, weight=data.shape[0])
    
        # Batch validation accuracy.
        batch_acc = acc_metric(pred_idx.cpu(), target.cpu())
    
        # Update progress bar description.
        step_status = status + f"\tLoss: {mean_metric.compute():.4f}, Acc: {acc_metric.compute():.4f}"
        prog_bar.set_description(step_status)
    
    test_loss = mean_metric.compute()
    test_acc = acc_metric.compute()
    
    prog_bar.close()
    
    return test_loss, test_acc