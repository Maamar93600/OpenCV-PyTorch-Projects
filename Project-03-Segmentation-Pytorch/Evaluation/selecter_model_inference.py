from librairie import *
from Utils.utils import selecte_model



def selecte_model_inference(dic_model, archi, backbone, run, device):
    model = selecte_model(dic_model, archi, backbone)

    path_load_weight = glob(os.path.join(TrainingConfig.LOG_DIR, archi, backbone, run, "*.tar"), recursive=True)[0]
    print(f"path model:{path_load_weight}")
    model.load_state_dict(torch.load(path_load_weight, map_location="cpu")["model"])
    model.to(device)
    model.eval()
    return model


