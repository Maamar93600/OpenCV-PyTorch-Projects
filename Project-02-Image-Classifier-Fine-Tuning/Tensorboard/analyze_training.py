mods=["effi_b0","resnet34","resnet50"]
dicto={}
for L in mods:
    dicto[L]={}
    for i in glob.glob(os.path.join(Config.LOG_DIR,L,"base*")):
        exp = os.path.basename(i)
        dicto[L][exp]={}
        for TAGS in ('Accuracy/Validation','Accuracy/Train','Loss/Validation','Loss/Train'):
            LOGGing=os.path.join(Config.LOG_DIR,L,exp)
            ea=EventAccumulator(LOGGing)
            ea.Reload()
            dicto[L][exp][TAGS]=[x.value for x in ea.Scalars(TAGS)]



fig, ax = plt.subplots(1, 2, figsize=(15, 5))

axe = {"Accuracy/Train":[0,"Train","Accuracy"],
       "Accuracy/Validation":[0,"Validation","Accuracy"],
       "Loss/Train":[1,"Train","Loss"],
       "Loss/Validation":[1,"Validation","Loss"]}


#for L in dicto:
for M,data in dicto.items():
    for exp,metrics in data.items():
        for tag,value in metrics.items():
            enu,lab,title=axe[tag]

            if lab=="Validation":
                line, = ax[enu].plot(value,linestyle="-",label=f"{M}-{exp} - {lab}")
                color=line.get_color()
            else:
                ax[enu].plot(value,linestyle="--",color=line.get_color(),label=f"{M}-{exp} - {lab}")
    
            ax[enu].set_title(title)
            ax[enu].grid()
            ax[enu].legend()
    


for M,data in dicto.items():
    for exp,metrics in data.items():
        TRAIN=dicto[M][exp]["Accuracy/Train"][-1]*100
        VALIDATION=dicto[M][exp]["Accuracy/Validation"][-1]*100
        
        eas_val=dicto[M][exp]["Accuracy/Validation"]
        best_epoch=eas_val.index(max(eas_val))+1
                       
        Max_Train=dicto[M][exp]["Accuracy/Train"][best_epoch]*100
        gap=TRAIN - VALIDATION
        gap_Max=Max_Train-max(eas_val)*100
        
        print(f"{M}-{exp}- best_epoch:{best_epoch} - Train:{TRAIN:.4f} - Validation:{VALIDATION:.4f} - gap:{gap:.4f} - Val_Accu_max:{max(eas_val)*100:.4f} - gap_Max:{gap_Max:.4f}")

#We observe that with Dropout regularization set to 0.2, increasing the RandAugment magnitude to 6 increased 
#the gap to 14.3088 compared with 13.2954 for magnitude 5.However, we also achieved a higher validation accuracy. 
#Therefore, we will generate the confusion matrix for the ResNet50 model with Dropout 0.2 and RandAugment magnitude 6.

#base :P10_F_LS_Ag +weight_decay,+scheduler(Cosine),discriminativeLR , (F-->fine=3 or 2  for resnet),LS=label smooth,Ag:RandAugmen
#resnet50
#base_dropout_vM6 (vM: Magnitude 6 )
#base_dropout_vM5 (vM: Magnitude 5 )
#base

#effib0
#base


# EXPERIMENTS
#
# BASE
# ├── ResNet50
# │   ├── Fine=3
# │   ├── Fine=3 + Dropout + RandAugment M5
# │   ├── Fine=3 + Dropout + RandAugment M6
# │   └── Fine=2 + Dropout + RandAugment M6  ← Final
# │
# └── EfficientNet-B0
#     └── Fine=3