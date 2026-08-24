class MONDATASET_TEST(torch.utils.data.Dataset):
   
    def __init__(self,data_root,file_csv):
        self.data_root=data_root
        self.dataframe=pd.read_csv(os.path.join(self.data_root,file_csv))
        #renvoi list des chemins des images
        self.imgs=[os.path.join(self.data_root,"images",f"{i}.jpg") for i in self.dataframe.iloc[:,0]]
                
        
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        
        self.transform=T.Compose([T.Resize((256,256)),
                                  T.CenterCrop(224),
                                  T.ToTensor(),
                                  T.Normalize(mean=mean, std=std)])
    def __len__(self):
        #nombre d'image
        return len(self.dataframe)
    
    def __getitem__(self,idx):

        image=Image.open(self.imgs[idx]).convert("RGB")
        image=self.transform(image)
        return image
