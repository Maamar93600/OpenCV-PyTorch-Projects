path_image="dataset/Test"
file_csv="test.csv"

#create dataset test
dataset_Test=MONDATASET_TEST(path_image,file_csv,transform=common_transforms)
#batch loader for test
batch_Test=torch.utils.data.DataLoader(dataset=dataset_Test,                                       
                                       batch_size=training_config.batch_size,
                                       shuffle=False)

#Mode eval
trained_model.eval()
#move Model to device
trained_model.to(device)
PREDICTION=[]

with torch.no_grad():
    
    for batch_input in batch_Test:    
        #move data same device to model
        data = batch_input.to(device) 
        output = trained_model(data)
        
        # recovers predicted class index for each sample
        pred = output.argmax(dim=1)
        PREDICTION.extend(pred.to('cpu'))
       