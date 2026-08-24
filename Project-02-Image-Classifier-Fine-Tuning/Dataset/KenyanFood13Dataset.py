class KenyanFood13Dataset(Dataset):
    """
    Dataset for the Kenyan Food 13-class classification task.

    CSV format:
        - First column: image identifier
        - Second column: class label
    """
    
    def __init__(self,data_root,file_csv,transform=False):
     
        self.transform=transform
        self.file_csv=file_csv
        self.data_root=data_root
        self.dataframe=pd.read_csv(os.path.join(self.data_root,file_csv))
 
        self.imgs=[os.path.join(self.data_root,"images",f"{i}.jpg") for i in self.dataframe.iloc[:,0]]
        self.label=sorted(self.dataframe.iloc[:,1].unique())
        self.classe_to_idx={C:idx for idx,C in enumerate(self.label)}
        # Compute normalized class weights for class imbalance
        P=(self.dataframe.iloc[:,1].value_counts().sort_index()/(self.dataframe.iloc[:,1].count().item())).values
        self.poids=torch.tensor((1/P)/(1/P).mean(),dtype=torch.float)
        
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        
        if self.transform:
            self.transform=T.Compose([T.Resize((256,256)),
                                      T.RandomCrop(224),
                                      T.RandomHorizontalFlip(),
                                      T.RandAugment(num_ops=5,magnitude=6),
                                      T.ToTensor(),
                                      T.RandomErasing(p=0.25),
                                      T.Normalize(mean=mean, std=std)])
        else:
            self.transform=T.Compose([T.Resize((256,256)),
                                            T.CenterCrop(224),
                                            T.ToTensor(),
                                            T.Normalize(mean=mean, std=std)])

    def __len__(self):
        
        return len(self.imgs)
    
    def __getitem__(self, idx):
        Images=Image.open(self.imgs[idx]).convert("RGB")
        label=self.dataframe.iloc[idx,1]

        image=self.transform(Images)            
            
        return image,self.classe_to_idx[label]
