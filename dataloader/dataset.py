import collections.abc
import numpy as np

from torch.utils.data import Dataset as _TorchDataset
from typing import Any, Callable, Optional, Sequence, Union
from torch.utils.data import Subset


class Dataset(_TorchDataset):
    """
    A generic data with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = data[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data, transform=None) -> None:
        """
        Args:
            data: input data to load and transform to generate data for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.transform: Any = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return self.transform(data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # data[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # data[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)



