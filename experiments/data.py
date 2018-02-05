import numpy as np
import random


class SeqDataset:
    def __init__(self, input_seq=None, target_seq=None):
        self.input_seq = input_seq
        self.target_seq = target_seq


class Datasets:
    def __init__(self,
                 train: SeqDataset=None,
                 dev: SeqDataset=None,
                 test: SeqDataset=None,
                 debug: SeqDataset=None):
        self.train = train
        self.dev = dev
        self.test = test
        self.debug = debug


class DatasetBatcher:
    def __init__(self, dataset: SeqDataset, batch_size):
        instance_no, length = np.shape(dataset.target_seq)
        assert batch_size <= instance_no

        self._dataset = dataset
        self._batch_size = batch_size
        self._batch_number = instance_no // self._batch_size
        self._generator = self._batcher()
        self._current_epoch = 0
        self._current_batch = 0

    def next_batch(self):
        return next(self._generator)

    def _batcher(self):
        while True:
            shuffle_array = list(range(0, self._batch_number)) * self._batch_size
            random.shuffle(shuffle_array)
            shuffle_partitions = [[] for _ in range(self._batch_number)]

            for i, value in enumerate(shuffle_array):
                shuffle_partitions[value].append(i)

            for i in range(self._batch_number):
                partition = shuffle_partitions[i]
                input_seq = self._dataset.input_seq[partition]
                target_seq = self._dataset.target_seq[partition]
                yield SeqDataset(input_seq, target_seq)

                self._current_batch += 1

            self._current_epoch += 1
            self._current_batch = 0


def load_single_dataset(filename):
    ret_y = []
    ret_x = []
    try:
        with open(filename, 'r') as f:
            next(f)         # skip the first line, expect comment by default
            for line in f:
                split = line.rstrip().split('\t')
                y = split[0]
                x = split[1]
                ret_x.append([int(num) for num in x.strip('[]').split(' ')])
                ret_y.append([int(num) for num in y.strip('[]').split(' ')])
        return SeqDataset(np.vstack(ret_x), np.vstack(ret_y))
    except FileNotFoundError:
        print('Cannot find' + filename)
        return SeqDataset()


def load_data(directory):
    ret = Datasets()
    for dataset in ["train", "dev", "test", "debug"]:
        filename = directory + dataset + '.txt'
        loaded_dataset = load_single_dataset(filename)
        setattr(ret, dataset, loaded_dataset)
    return ret


if __name__ == "__main__":

    print("Loading data...")

    datasets = load_data('./experiments/data/add/configX/')

    shape = np.shape(datasets.train.input_seq)
    print(" train", "- instance no:", shape[0], ", seq len:", (shape[1] - 2)//2)
    shape = np.shape(datasets.dev.input_seq)
    print(" dev", "- instance no:", shape[0], ", seq len:", (shape[1] - 2)//2)
    shape = np.shape(datasets.test.input_seq)
    print(" test", "- instance no:", shape[0], ", seq len:", (shape[1] - 2)//2)
    shape = np.shape(datasets.debug.input_seq)
    print(" debug", "- instance no:", shape[0], ", seq len:", (shape[1] - 2)//2)

    debug_batcher = DatasetBatcher(datasets.debug, batch_size=5)
    for _ in range(4):
        batch = debug_batcher.next_batch()
        print('--' * 50)
        print("current epoch", debug_batcher._current_epoch)
        print("current batch", debug_batcher._current_batch)
        print(batch.input_seq)
        print(batch.target_seq)
