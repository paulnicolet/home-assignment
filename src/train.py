from pathlib import Path

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from .tomatoes_dataset import TomatoesDataset

EPOCHS = 21
BATCH_SIZE = 32
LR = 0.001


def train(imgs_root, annotation_path, label_mapping_path, device):
    """Train a new model from the entire datast.

    Args:
        imgs_root (str): The path to the image directory.
        annotation_path (str): The path to the annotations file.
        label_mapping_path (str): The path to the label mapping file.
        device (str): "cuda" or "cpu".

    Returns:
        ResNet: A Pytorch resnet.
    """
    # Fetch data
    dataset = TomatoesDataset(
        Path(imgs_root),
        Path(annotation_path),
        Path(label_mapping_path),
        sampling='over'
    )
    loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

    # Init model and optimizer
    model = new_model()
    criterion = BCEWithLogitsLoss()
    if device == 'cuda':
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train
    model.train()
    for epoch in tqdm(range(EPOCHS)):
        for inputs, labels in loader:
            if device == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()

            inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def new_model():
    """Build a fresh new pre-trained ResNet.

    Returns:
        ResNet: A Pytorch ResNet.
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    return model
