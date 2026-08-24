def train(DEVICE,Config,model,optimizer,train_loader,epoch_idx,total_epochs):
    
    # change model in training mode
    model.train()
    model=model.to(DEVICE)
    
    acc_metric = MulticlassAccuracy(num_classes=Config.NUM_CLASSES, average="micro")
    mean_metric = MeanMetric()
    
    status = f"Train:\t{bold}Epoch: {epoch_idx}/{total_epochs}{reset}"
    
    prog_bar = tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    prog_bar.set_description(status)
    
    for data, target in prog_bar:
    
        # Send data and target to appropriate device.
        data, target = data.to(DEVICE), target.to(DEVICE)
    
        # Reset parameters gradient to zero.
        optimizer.zero_grad()
    
        # Forward pass to the model.
        output = model(data)
        
        # Cross Entropy loss
        
        loss = F.cross_entropy(output, target,weight=Config.Poids.to(DEVICE),label_smoothing=0.1)
    
        # Find gradients w.r.t training parameters.
        loss.backward()
    
        # Update parameters using gradients.
        optimizer.step()
    
        # Batch Loss.
        batch_loss = mean_metric(loss.item(), weight=data.shape[0])
    
        # Get probability score using softmax.
        prob = F.softmax(output, dim=1)
    
        # Get the index of the max probability.
        pred_idx = prob.detach().argmax(dim=1)
    
        # Batch accuracy.
        batch_acc = acc_metric(pred_idx.cpu(), target.cpu())
    
        # Update progress bar description.
        step_status = status + f"\tLoss: {mean_metric.compute():.4f}, Acc: {acc_metric.compute():.4f}"
        prog_bar.set_description(step_status)
    
    epoch_loss = mean_metric.compute()
    epoch_acc = acc_metric.compute()
    
    prog_bar.close()
    
    return epoch_loss, epoch_acc