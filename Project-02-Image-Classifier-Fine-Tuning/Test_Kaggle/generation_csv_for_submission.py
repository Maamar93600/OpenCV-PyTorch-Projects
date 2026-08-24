dataset_Test=MONDATASET_TEST("data","test.csv")
classe=KenyanFood13Dataset("data","train.csv").label

#recupère ID
ID=[os.path.splitext((os.path.basename(imgs)))[0] for imgs in dataset_Test.imgs]
#prediction from Valid set
PRED_label=[classe[i] for i in PREDICTION]

d={'ID':ID,'CLASS':PRED_label}
DATAFRAME=pd.DataFrame(data=d)
DATAFRAME.to_csv("Maamarv2_Sumission.csv",index=False)