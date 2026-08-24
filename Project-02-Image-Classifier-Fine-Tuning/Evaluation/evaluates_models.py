device=("cuda" if torch.cuda.is_available() else "cpu")

#load Model
LISTE=[("resnet50","base_drop_M6"),("resnet50","base_F2_drop_M6"),("resnet50","base_drop_M5")]
for MODELS_,EXP_ in LISTE:
    
    get_Model=load_model(MODELS_,EXP_)
    
    _, valid_loader = get_data(
        batch_size=Config.batch_size,
        data_root=Config.data_root,
        num_workers=Config.num_workers)
    
    
    CM=MulticlassConfusionMatrix(num_classes=len(valid_loader.dataset.dataset.label),normalize='true').to(device)
    classe=valid_loader.dataset.dataset.label
    
    
    get_Model.eval()
    get_Model.to(device)
    PRED=[]
    TARGET=[]
    with torch.no_grad():
        
        for batch_input,label in valid_loader:        
            data,label = batch_input.to(device),label.to(device)    
            output = get_Model(data)
            
            pred = output.argmax(dim=1)
            PRED.extend(pred.cpu().numpy())
            TARGET.extend(label.cpu().numpy())
            CM.update(pred,label)

    print(classification_report(TARGET,PRED,target_names=classe))
    Matrice_Confusion(CM.compute().to("cpu"),valid_loader.dataset.dataset.label)