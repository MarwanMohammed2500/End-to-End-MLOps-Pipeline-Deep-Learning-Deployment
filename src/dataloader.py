from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(root:str="./data", batch_size:int=32, download:bool=True):
    """
    get_dataloaders is desgined to import the FashionMNIST dataset (train and test) and load them into DataLoaders properly

    Arguments:
    root:str="data", Where the data is stored
    batch_size:int=32, Batch size used in the DataLoader
    download:bool=True, Whether or not to download the dataset

    Returns:
    train_dataloader, test_dataloader: torch.utils.data.DataLoader, The train and test DataLoaders
    """

    transform = transforms.ToTensor() # Turn the images into tensors

    #                                                   === Get the train and test Datasets ===
    # Training
    train_data = datasets.FashionMNIST(root=root,
                                    transform=transform,
                                    train=True, # Get the training data
                                    download=download, # Download the dataset
                                    target_transform=None # Don't apply any transformations on the targets
                                    )
    # Testing
    test_data = datasets.FashionMNIST(root=root,
                                    transform=transform,
                                    train=False, # Get the training data
                                    download=download, # Download the dataset
                                    target_transform=None # Don't apply any transformations on the targets
                                    )

    #                                                   === Set up the DataLoaders ===
    # Train DataLoader
    train_dataloader = DataLoader(dataset=train_data, # Use `train_data` as the `Dataset` for this `DataLoader`
                                batch_size=batch_size,
                                shuffle=True # Shuffle the data for training purposes
                                )

    # Test DataLoader
    test_dataloader = DataLoader(dataset=test_data, # Use `test_data` as the `Dataset` for this `DataLoader`
                                batch_size=batch_size,
                                shuffle=False # Don't shuffle the data
                                )
    return train_dataloader, test_dataloader