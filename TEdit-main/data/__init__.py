from torch.utils.data import DataLoader

from .synthetic_finetune import SyntheticDataset
from .synthetic_pretrain import SyntheticPretrainDataset

datasets = {
    "synthetic": SyntheticDataset,
    "synthetic_pretrain": SyntheticPretrainDataset,
}


class EditDataset:
    def __init__(self, configs):
        self.configs = configs
        self.dataset = datasets[configs["name"]](**configs)

    def get_loader(self, split, batch_size, shuffle=True, num_workers=1, include_self=False):
        loader = DataLoader(
            dataset=self.dataset.get_split(split, include_self), 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers)
        return loader

    @property
    def ctrl_attr_ops(self):
        return self.dataset.ctrl_attr_ops

    @property
    def side_attr_ops(self):
        return self.dataset.side_attr_ops

    # new
    @property
    def num_attr_ops(self):
        return self.dataset.attr_n_ops

    @property
    def ctrl_attr_ids(self):
        return self.dataset.ctrl_attr_ids

    @property
    def side_attr_ids(self):
        return self.dataset.side_attr_ids