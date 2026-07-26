def image_preprocess_transforms(img_size):
    preprocess = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor()           
        ]
    )

    return preprocess



def image_common_transforms(img_size=(256, 256), mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    #preprocess = image_preprocess_transforms(img_size)

    common_transforms = transforms.Compose(
        [
            transforms.Resize(img_size),
            #transforms.CenterCrop((224,224)),
            #transforms.RandomResizedCrop(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2,contrast=0.2)],p=0.3), 
            transforms.RandomApply([transforms.RandomAffine(degrees=(-10,10),translate=(0.1,0.1),scale=(0.90,1.1))],p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    return common_transforms


