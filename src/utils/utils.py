import os
import torch
import logging as lg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model_weights, model_name, dir="./checkpoints/"):
    """Save PyTorch model weights
    Args:
        model_weights (Dict): model stat_dict
        model_name (str): name_of_the_model.pth
    """
    lg.debug("Saving checkpoint...")
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    torch.save(model_weights, os.path.join(dir, model_name))

# @utils.tensorfy(0, 1, tensor_klass=torch.LongTensor)
def filter_labels(y, labels):
    """Utility used to create a mask to filter values in a tensor.

    Args:
        y (list, torch.Tensor): tensor where each element is a numeric integer
            representing a label.
        labels (list, torch.Tensor): filter used to generate the mask. For each
            value in ``y`` its mask will be "1" if its value is in ``labels``,
            "0" otherwise".

    Shape:
        y: can have any shape. Usually will be :math:`(N, S)` or :math:`(S)`,
            containing `batch X samples` or just a list of `samples`.
        labels: a flatten list, or a 1D LongTensor.

    Returns:
        mask (torch.ByteTensor): a binary mask, with "1" with the respective value from ``y`` is
        in the ``labels`` filter.

    Example::

        >>> a = torch.LongTensor([[1,2,3],[1,1,2],[3,5,1]])
        >>> a
         1  2  3
         1  1  2
         3  5  1
        [torch.LongTensor of size 3x3]
        >>> classification.filter_labels(a, [1, 2, 5])
         1  1  0
         1  1  1
         0  1  1
        [torch.ByteTensor of size 3x3]
        >>> classification.filter_labels(a, torch.LongTensor([1]))
         1  0  0
         1  1  0
         0  0  1
        [torch.ByteTensor of size 3x3]
    """
    mapping = torch.zeros(y.size()).byte()

    for label in labels:
        mapping = mapping | y.eq(label)

    return mapping
