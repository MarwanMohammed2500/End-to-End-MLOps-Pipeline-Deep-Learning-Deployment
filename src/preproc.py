def merge_labels(labels):
    """
    merges the "Shirt" and "T-Shirt/top" classes together to reduce ambiguity

    Arguments:
    labels:torch.Tensor, The labels `Tensor`
    """
    new_labels = labels.clone()
    new_labels[new_labels == 6] = 0
    new_labels[new_labels >= 7] = new_labels[new_labels >= 7] - 1
    return new_labels