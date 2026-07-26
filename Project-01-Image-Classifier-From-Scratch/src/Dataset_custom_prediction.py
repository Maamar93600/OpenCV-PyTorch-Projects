from PIL import Image



class MONDATASET_TEST(torch.utils.data.Dataset):
   
    def __init__(self,chemin_image,fichier_csv,transform=None):
        self.path=chemin_image
        self.dataframe=pd.read_csv(fichier_csv)
        #renvoi list des chemins des images
        self.imgs=[os.path.join(self.path,self.dataframe.iloc[i,0]) for i in range(len(self.dataframe))]
        self.transform=transform

    def __repr__(self):
        return (
                f"Dataset Test\n"
                f"    Number of datapoints: {len(self.dataframe)}\n"
                f"    Root location: {self.path}")

     
    def __len__(self):
        #nombre d'image
        return len(self.dataframe)
    
    def __getitem__(self,idx):

        image=Image.open(os.path.join(self.path,self.dataframe.iloc[idx,0])).convert("RGB")
        if self.transform:
            image=self.transform(image)#transform image PIL to tensor pre-process
         
            
        return image
