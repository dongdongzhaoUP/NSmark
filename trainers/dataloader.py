import torch
from torch.utils.data import DataLoader


def get_dict_dataloader(dataset, batch_size, drop_last=False):
    dataloader = {}
    for split in dataset.keys():
        dataloader[split] = get_dataloader(dataset[split], batch_size, drop_last)
    return dataloader


def get_dataloader(dataset, batch_size, drop_last, shuffle=True):
    def collate_fn(batch_example):
        batch = {
            "text_a": [e.text_a for e in batch_example],
            "text_b": [e.text_b for e in batch_example],
            "label": [e.label for e in batch_example],
            "embed": [e.embed for e in batch_example],
            "poison_label": [e.poison_label for e in batch_example],
        }
        return batch

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
