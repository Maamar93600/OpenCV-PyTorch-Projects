def get_data(batch_size,data_root,num_workers=2):

    #indice for train dataset and valid dataset
    DATASET=KenyanFood13Dataset(data_root,"train.csv")
    train_test_split=[0.8,0.2]
    generator=torch.Generator().manual_seed(42)
    train_set,val_set=random_split(dataset=DATASET,lengths=train_test_split,generator=generator)
    train_indice=train_set.indices
    valid_indice=val_set.indices

    #dataset train/valid transform here TRAIN_DATASET and VALID_DATASET see same data but transformation is different
    TRAIN_=KenyanFood13Dataset(data_root,"train.csv",transform=True)
    #VALID_=KenyanFood13Dataset("data","train.csv",transform=False)

    #create really dataset for train and valid
    TRAIN_DATASET=Subset(TRAIN_,train_indice)
    VALID_DATASET=Subset(DATASET,valid_indice)
    
    #create Dataloader     
    Train_loader=DataLoader(dataset=TRAIN_DATASET,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
    Valid_loader=DataLoader(dataset=VALID_DATASET,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    return Train_loader, Valid_loader