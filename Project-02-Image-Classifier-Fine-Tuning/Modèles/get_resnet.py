#from torchvision.models import resnet18,resnet34,resnet50
#resnet 14,38,50 same layer 
def get_resnet(resnet_model_name="resnet18",num_classes=10,poids="DEFAULT",fine_tune_start=5,drop=0):

    model=getattr(models,resnet_model_name)(weights=poids)
    
    if poids:
        for param in model.parameters():
            param.requires_grad = False

    if poids and fine_tune_start <= 1:
        for param in model.layer1.parameters():
            param.requires_grad = True

    if poids and fine_tune_start <= 2:
        for param in model.layer2.parameters():
            param.requires_grad = True

    if poids and fine_tune_start <= 3:
        for param in model.layer3.parameters():
            param.requires_grad = True

    if poids and fine_tune_start <= 4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    model_fc_in_features = model.fc.in_features

    model.fc = nn.Sequential(nn.Dropout(p=drop),
                             nn.Linear(in_features=model_fc_in_features, out_features=num_classes))

    return model