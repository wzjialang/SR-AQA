from typing import Optional, Sequence
import os
from image_regression import ImageRegression


class TOE(ImageRegression):
    """`DSprites <https://github.com/deepmind/dsprites-dataset>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'C'``: Color, \
            ``'N'``: Noisy and ``'S'``: Scream.
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        factors (sequence[str]): Factors selected. Default: ('scale', 'position x', 'position y').
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            color/
                ...
            noisy/
            scream/
            image_list/
                color_train.txt
                noisy_train.txt
                scream_train.txt
                color_test.txt
                noisy_test.txt
                scream_test.txt
    """
    image_list = {
        "S": "simulated_data_frames",
        "R": "real_cases_data_frames",
    }
    FACTORS = ('class', 'CP', 'GI')

    def __init__(self, root: str, task: str, split: Optional[str] = 'train',
                 factors: Sequence[str] = ('class', 'CP', 'GI'),
                 download: Optional[bool] = True, target_transform=None, **kwargs):
        assert task in self.image_list
        assert split in ['train', 'test']
        for factor in factors:
            assert factor in self.FACTORS

        factor_index = [self.FACTORS.index(factor) for factor in factors]

        if target_transform is None:
            target_transform = lambda x: x[list(factor_index)]
        else:
            target_transform = lambda x: target_transform(x[list(factor_index)])

        data_list_file = os.path.join(root, "{}.csv".format(self.image_list[task]))

        super(TOE, self).__init__(root, factors, data_list_file=data_list_file, target_transform=target_transform, **kwargs)

