#load model "resnet50" for best exp : base_drop_M6
MODELS_="resnet50"
EXP_="base_F2_drop_M6"
get_Model=load_model(MODELS_,EXP_)


#device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#batch loader for test
batch_Test=get_data_TEST(Config.batch_size,Config.data_root,file_csv="test.csv",num_workers=Config.num_workers)

#Mode eval
get_Model.eval()
#move Model to device
get_Model.to(device)
PREDICTION=[]

with torch.no_grad():
    
    for batch_input in batch_Test:    
        #move data same device to model
        data = batch_input.to(device) 
        output = get_Model(data)
        
        # recovers predicted class index for each sample
        pred = output.argmax(dim=1)
        PREDICTION.extend(pred.cpu().numpy())