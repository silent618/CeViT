import pandas as pd

from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from .dataset import Dataset
from .transforms import LoadImage, CopyDict


def get_loader(dataframes_path, batch_size=8, num_workers=0, root_path='', ):

    # Prepare data sample pre-processing transforms
    transforms = Compose([
        CopyDict(),
        LoadImage(root_path=root_path),
    ])

    data = []
    dataframe = pd.read_csv(dataframes_path)

    for i in range(len(dataframe)):
        data_i = dataframe.loc[i, :].to_dict()

        data.append(data_i)

    train_dataset = Dataset(data=data, transform=transforms)

    # Set dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader


def get_datasets(dataframes_path, root_path=''):
    # Prepare data sample pre-processing transforms
    transforms = Compose([
        CopyDict(),
        LoadImage(root_path=root_path),
    ])

    data = []
    dataframe = pd.read_csv(dataframes_path)

    for i in range(len(dataframe)):
        data_i = dataframe.loc[i, :].to_dict()

        data.append(data_i)

    dataset = Dataset(data=data, transform=transforms)

    return dataset


"""dataloader = get_loader(dataframes_path="../simulation.csv", batch_size=24, root_path='D:/Code/OUCopula')

a = iter(dataloader)
b = next(a)
c = 1"""
