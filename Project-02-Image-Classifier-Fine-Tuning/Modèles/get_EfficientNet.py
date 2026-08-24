def get_EfficientNet(EFFI_model_name="efficientnet_b0",num_classes=10,poids="DEFAULT",fine_tune_start_Effi=5):

    model=getattr(models,EFFI_model_name)(weights=poids)
    #dict_keys(['0', '1', '2', '3', '4', '5', '6', '7', '8'])
    if poids:
        for param in model.parameters():
            param.requires_grad = False

    if poids and fine_tune_start_Effi <= 1:
        for param in model.features[5].parameters():
            param.requires_grad = True

    if poids and fine_tune_start_Effi <= 2:
        for param in model.features[6].parameters():
            param.requires_grad = True

    if poids and fine_tune_start_Effi <= 3:
        for param in model.features[7].parameters():
            param.requires_grad = True

    if poids and fine_tune_start_Effi <= 4:
        for param in model.features[8].parameters():
            param.requires_grad = True

    model_fc_in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(in_features=model_fc_in_features, out_features=num_classes)

    return model