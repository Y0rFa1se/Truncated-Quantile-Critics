import pytorch_lightning as L
from torch.utils.data import IterableDataset, DataLoader

class RLDataset(IterableDataset):
    def __init__(self, buffer, batch_size=256, sample_per_epoch=1000):
        self.buffer = buffer
        self.batch_size = batch_size
        self.sample_per_epoch = sample_per_epoch

    def __iter__(self):
        for _ in range(self.sample_per_epoch):
            yield self.buffer.sample(self.batch_size)

def get_dataloader(buffer, batch_size=256, sample_per_epoch=1000):
    dataset = RLDataset(buffer, batch_size, sample_per_epoch)

    return DataLoader(dataset, batch_size=None)