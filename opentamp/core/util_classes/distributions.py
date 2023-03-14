import numpy as np
import torch

class Distribution(np.ndarray):
    """
    Distribution tracks random quantities for belief-space planning.
    """
    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("Override this.")


class Gaussian(Distribution):
    """
    The vector class.
    """
    def __new__(cls, vec):
        if type(vec) is str:
            if not vec.endswith(")"):
                vec += ")"
            vec = eval(vec)
        obj = torch.distributions.normal.Normal(torch.tensor([vec[0]]), torch.tensor([vec[1]]))
        assert len(vec) == 2  # only two parameters expected
        print(obj)
        return obj
