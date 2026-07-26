class MyModel(nn.Module):
    
    def __init__(self,in_channels=3, num_classes=TrainingConfig.num_classes, space_drop=0.3, drop=0.2):
        super().__init__()
        self._body=nn.Sequential(
            
            Conv2dNormActivation(in_channels=in_channels,out_channels=64,kernel_size=3,padding='same'),
            Conv2dNormActivation(in_channels=64,out_channels=64,kernel_size=3,padding='same'),
            nn.MaxPool2d(2),
            Conv2dNormActivation(in_channels=64,out_channels=128,kernel_size=3,padding='same'),
            Conv2dNormActivation(in_channels=128,out_channels=128,kernel_size=3,padding='same'),
            nn.MaxPool2d(2),
            Conv2dNormActivation(in_channels=128,out_channels=256,kernel_size=3,padding='same'),
            Conv2dNormActivation(in_channels=256,out_channels=256,kernel_size=3,padding='same'),
            nn.Dropout2d(space_drop),
            nn.MaxPool2d(2),
            
            Conv2dNormActivation(in_channels=256,out_channels=512,kernel_size=3,padding='same'),
            Conv2dNormActivation(in_channels=512,out_channels=512,kernel_size=3,padding='same'),
            nn.Dropout2d(space_drop),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(output_size=(5,5)))
        
        
        self._head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=5*5*512,out_features=64), 
            nn.ReLU(),
            nn.Dropout(p=drop),
            
            nn.Linear(in_features=64,out_features=32), 
            nn.ReLU(),
            nn.Dropout(p=drop),
              
            nn.Linear(in_features=32,out_features=TrainingConfig.num_classes))
                      
    def forward(self,x):
        x=self._body(x)
        x=self._head(x)
        return x        
            
                      
    
