from .train import new_model
from .tomatoes_dataset import TomatoesDataset
import torch

def has_tomatoes(img_path, checkpoint_path):
    model = _load_model(checkpoint_path)
    tensor = TomatoesDataset.prepare_single_image(img_path)
    output = model(tensor.unsqueeze(0))
    print(output)
    print(torch.max(output, dim=1))
    label_index = torch.max(output, dim=1)[1][0]
    print(label_index)
    return bool(label_index)

def _load_model(checkpoint_path):
    model = new_model()
    model.load_state_dict(torch.load(checkpoint_path))
    return model