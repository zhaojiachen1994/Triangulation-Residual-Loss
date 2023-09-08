import numpy as np
from torch.utils.data import Dataset

from mmpose.datasets.builder import DATASETS, build_dataset


@DATASETS.register_module()
class CrossDomain2d3dDataset(Dataset):
    """
    Mix dataset for cross domain 2d-3d pose estimation task.

    The dataset combines data from two datasets:
    a 2d source dataset containing 2d ground truth;
    a 3d target dataset without ground truth labels
    """

    def __init__(self, source_data, target_data, test_mode=False):
        super().__init__()
        if source_data is not None:
            self.source_dataset = build_dataset(source_data)
        self.target_dataset = build_dataset(target_data)

    @property
    def with_source_dataset(self):
        return hasattr(self, 'source_dataset')

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.target_dataset)

    def __getitem__(self, idx):
        results = {}
        target_data = self.target_dataset[idx]
        results['target_data'] = target_data

        if self.with_source_dataset:
            rand_ind = np.random.randint(0, len(self.source_dataset))
            source_data = self.source_dataset[rand_ind]
            results['source_data'] = source_data
        return results

    def evaluate(self, results, res_folder=None, metric='mpjpe', **kwargs):
        return self.target_dataset.evaluate(results, res_folder=None, metric='mpjpe', **kwargs)
