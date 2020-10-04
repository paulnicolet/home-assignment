import torch

from .tomatoes_dataset import TomatoesDataset
from .train import new_model


def has_tomatoes(img_path, checkpoint_path):
    """Returns True if the given image contains traces of tomato.

    Args:
        img_path (str): The image path.
        checkpoint_path (str): The model checkpoint path.

    Returns:
        bool: True if the image contains traces of tomato, False otherwise.
    """
    model = _load_model(checkpoint_path)
    tensor = TomatoesDataset.prepare_single_image(img_path)
    output = model(tensor.unsqueeze(0))
    label_index = torch.max(output, dim=1)[1][0]
    return bool(label_index)


def _load_model(checkpoint_path):
    model = new_model()
    model.load_state_dict(torch.load(checkpoint_path))
    return model
