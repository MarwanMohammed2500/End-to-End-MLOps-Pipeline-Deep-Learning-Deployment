import matplotlib.pyplot as plt
import random
from utils import set_seed

def plot_images(images_batch, classes, labels_batch, size=(3,3), figsize=(10,10)):
    """
    Plots random images from the dataset

    Arguments:
    images:torch.Tensor, the images
    labels:torch.Tensor, the labels
    classes:list, the classes
    size:tuple=(3,3) the size of the grid (nrows, ncols)
    """
    set_seed()
    nrows, ncols = size
    random_idx = random.sample(range(0, len(images_batch)), nrows*ncols)
    plt.figure(figsize=figsize)
    for img in range(nrows*ncols):
        plt.subplot(nrows, ncols, img+1)
        plt.imshow(images_batch[random_idx[img]].squeeze().cpu(), cmap="Grays")
        plt.title(classes[labels_batch[random_idx[img]].item()])
        plt.axis(False)

def plot_predictions(images_batch, labels_batch, predicted_labels, size=(3,3), figsize=(10,10)):
    """
    Plots random images from the dataset

    Arguments:
    images:torch.Tensor, the images
    labels:torch.Tensor, the labels
    classes:list, the classes
    predicted_labels:torch.Tensor, the predicted labels
    size:tuple=(3,3), the size of the grid (nrows, ncols)
    """
    set_seed()
    nrows, ncols = size
    random_idx = random.sample(range(0, len(images_batch)), nrows*ncols)
    plt.figure(figsize=figsize)
    for img in range(nrows*ncols):
        plt.subplot(nrows, ncols, img+1)
        plt.imshow(images_batch[random_idx[img]].squeeze().cpu(), cmap="Grays")
        true_class = labels_batch[random_idx[img]]
        pred_class = predicted_labels[random_idx[img]]

        if true_class == pred_class:
            plt.title(f"True Class: {true_class} | Predicted Class: {pred_class}", c="green")
        else:
            plt.title(f"True Class: {true_class} | Predicted Class: {pred_class}", c="red")
        plt.axis(False)