from librairie import *



def find_class_weight(
        device,
        path_train,
        number_class
):

    pixel_weight = torch.zeros(
        number_class,
        dtype=torch.float32
    )

    classe_pixel = torch.zeros(
        number_class,
        dtype=torch.float32
    )

    for M in glob(path_train):

        tensor_M = torch.tensor(
            np.array(Image.open(M)),
            dtype=torch.long
        )

        unique, counts = torch.unique(
            tensor_M,
            return_counts=True
        )

        classe_pixel[unique] += counts

    classe_pixel_frequence = (
        classe_pixel / classe_pixel.sum()
    )

    # 1 / frequency
    # pixel_weight = 1 / classe_pixel_frequence

    # 1 / sqrt(frequency)
    pixel_weight = 1 / torch.sqrt(
        classe_pixel_frequence
    )

    # Median Frequency Balancing
    # pixel_weight = (
    #     classe_pixel_frequence.median()
    #     / classe_pixel_frequence
    # )

    pixel_weight = (
        pixel_weight / pixel_weight.mean()
    )

    return pixel_weight.to(device), classe_pixel_frequence
