from librairie import *

def get_default_device():
    gpu_available = torch.cuda.is_available()
    return torch.device('cuda' if gpu_available else 'cpu'), gpu_available

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def create_run_dir(checkpoint_dir):
    # Create a new checkpoint directory every time.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    try:
        num_versions = [int(i.split("_")[-1]) for i in os.listdir(checkpoint_dir) if "Run" in i]
        version_num = max(num_versions) + 1
    except:
        version_num = 0

    version_dir = os.path.join(checkpoint_dir, "Run_" + str(version_num))
    os.makedirs(version_dir)
    run_experience=os.path.split(version_dir)[-1].replace('Run','Exp')

    print(f"Checkpoint directory: {version_dir}")
    return version_dir,run_experience


def create_dir_model(directory, model):
    path = os.path.join(directory, model)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def selecte_model(dic_model, archi=TrainingConfig.Deeplab, backbone="resnet50"):
    model = f"{archi}_{backbone}"
    if model in dic_model.keys():
        print(f"Load model : {model}")
        return dic_model[model]
    else:
        print("Model doesn\'t exist")




