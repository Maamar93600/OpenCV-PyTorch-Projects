from librairie import *

dic_model={}
def add_model(model):
    if model.__class__.__name__ not in dic_model:
        name_model=f"{model.__class__.__name__}_{model.config['encoder_name']}"
        dic_model[name_model]=model


# 1. U-Net ResNet50
model_Unet_resnet50=smp.Unet(encoder_name ='resnet50',
                         encoder_weights='imagenet',
                        encoder_depth=5,
                        decoder_channels= (256, 128, 64,32,16),
                        in_channels=3,
                         classes=12)

print(summary(model_Unet_resnet50,input_size=(1,3,256,256),row_settings=["var_names"]))



# 2. DeepLabV3+ ResNet101
model_dlv3plus_resnet101 = smp.DeepLabV3Plus(encoder_name='resnet101',
                                             encoder_weights='imagenet',
                                             encoder_output_stride=16,
                                             decoder_channels=256,
                                             decoder_atrous_rates=(6, 12, 18),
                                             in_channels=3,
                                             classes=12)

print(summary(model_dlv3plus_resnet101,input_size=(1,3,384,384),row_settings=["var_names"]))



# 3. DeepLabV3+ ResNet50
model_dlv3plus_resnet50=smp.DeepLabV3Plus(encoder_name='resnet50',
                                 encoder_weights= 'imagenet',
                                 encoder_output_stride=16,
                                 decoder_channels= 256,
                                 decoder_atrous_rates= (6, 12, 18),
                                 in_channels = 3,
                                 classes= 12)

print(summary(model_dlv3plus_resnet50,input_size=(1,3,384,384),row_settings=["var_names"]))



# Ajout des modèles au dictionnaire
add_model(model_Unet_resnet50)
add_model(model_dlv3plus_resnet50)
add_model(model_dlv3plus_resnet101)

dic_model.keys()




