import torch


class FastDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.i = 0


    def __iter__(self):
        # TODO: Handle shuffling? The dataset is inherently unordered right now
        self.i = 0
        return self


    def __next__(self):
        if self.i >= len(self.dataset) // self.batch_size:
            raise StopIteration

        batch = self.dataset.get_batch(slice(self.i, self.i+self.batch_size))
        self.i += self.batch_size
        return batch


    def __len__(self):
        return len(self.dataset) // self.batch_size

