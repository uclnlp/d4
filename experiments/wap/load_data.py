import numpy as np
import random

vocab = {"PAD": 0, "SOS": 1, "EOS": 2}
id_to_sym = ["PAD", "SOS", "EOS"]

target_vocab = {}
target_id_to_sym = []


class ArithmeticDataset:
    def __init__(self, ids, seq_length, number_positions, numbers, target_values):
        self.tokens = ids
        self.seq_lengths = seq_length
        self.num_pos = number_positions
        self.nums = numbers
        self.targets = target_values


class ArithmeticData:
    def __init__(self, train, dev, test, debug):
        self.train = train
        self.dev = dev
        self.test = test
        self.debug = debug


def get_max_length(directory):
    max_seq_length = 0
    max_num_args = 0
    for corpus in ["train", "dev", "test"]:
        f = open(directory + corpus + ".txt", "r")
        for line in f.readlines():
            target, story = line.split("\t")
            length = len(story.split(" "))
            if length > max_seq_length:
                max_seq_length = length

            num_args = len([x for x in story.split(" ") if x.isnumeric()])
            if num_args > max_num_args:
                max_num_args = num_args

    return max_seq_length, max_num_args


def load_corpus(path, max_seq_length, max_num_args):
    """
    :param path: path to CC corpus
    :param max_seq_length: maximum sequence length
    :param max_num_args: maximum number of numbers in sequence
    :return:
        ids, seq_length, number_positions, numbers, target

        ids: [N x max_seq_length]
        seq_length: [N]
        number_positions: [N x max_num_args]
        numbers: [N x max_num_args]
        target: [N]
    """
    f = open(path, "r")

    ids = []
    seq_length = []
    number_positions = []
    numbers = []
    target = []

    for line in f.readlines():
        le_target, story = line.split("\t")
        tokenized_story = story.split(" ")
        # # last token always ends with '?'
        # tokenized_story[-1] = tokenized_story[-1][:-1]
        # tokenized_story.append("?")

        mapped = np.zeros(max_seq_length + 2)
        mapped[0] = vocab["SOS"]
        i = 1

        local_numbers = np.zeros(max_num_args)
        local_number_positions = np.zeros(max_num_args)

        number_counter = 0

        for token in tokenized_story:
            if token != "":
                if token not in vocab:
                    vocab[token] = len(id_to_sym)
                    id_to_sym.append(token)
                mapped[i] = vocab[token]
                if token.isnumeric():
                    local_numbers[number_counter] = float(token)
                    local_number_positions[number_counter] = i
                    number_counter += 1
                i += 1
        mapped[i] = vocab["EOS"]

        ids.append(mapped)
        seq_length.append(len(tokenized_story)+2)
        number_positions.append(local_number_positions)
        numbers.append(local_numbers)
        target.append(int(float(le_target)))

    return ArithmeticDataset(np.vstack(ids),
                             np.asarray(seq_length),
                             np.vstack(number_positions),
                             np.vstack(numbers),
                             np.vstack(target))


def load_all_datasets():
    print('Loading data')
    MAX_SEQ_LENGTH, MAX_ARGS = get_max_length("./data/wap/")
    train = load_corpus("./data/wap/train.txt", MAX_SEQ_LENGTH, MAX_ARGS)
    train_vocab = vocab.copy()
    dev = load_corpus("./data/wap/dev.txt", MAX_SEQ_LENGTH, MAX_ARGS)
    test = load_corpus("./data/wap/test.txt", MAX_SEQ_LENGTH, MAX_ARGS)
    debug = load_corpus("./data/wap/debug.txt", MAX_SEQ_LENGTH, MAX_ARGS)
    print('Loading complete')

    return ArithmeticData(train, dev, test, debug), MAX_SEQ_LENGTH, MAX_ARGS, len(vocab), vocab


class ArithmeticDatasetBatcher(object):
    def __init__(self, dataset: ArithmeticDataset, batch_size):
        assert batch_size <= len(dataset.tokens), "batch_size cannot be larger than dataset size"
        self._dataset = dataset
        self.batch_size = batch_size
        self.batch_number = len(self._dataset.tokens) // batch_size
        self._generator = self._batcher()
        # counters, for funsies
        self._current_epoch = 0
        self._current_batch = 0

    def next_batch(self):
        return next(self._generator)

    def _batcher(self):
        while True:
            shuffle_array = list(range(0, self.batch_number)) * self.batch_size
            random.shuffle(shuffle_array)
            shuffle_partitions = [[] for _ in range(self.batch_number)]

            for i, value in enumerate(shuffle_array):
                shuffle_partitions[value].append(i)

            if self.batch_number == 1:
                random.shuffle(shuffle_partitions[0])

            for i in range(self.batch_number):
                partition = shuffle_partitions[i]
                tokens = self._dataset.tokens[partition]
                seq_lengths = self._dataset.seq_lengths[partition]
                num_pos = self._dataset.num_pos[partition]
                nums = self._dataset.nums[partition]
                targets = self._dataset.targets[partition]
                yield ArithmeticDataset(tokens, seq_lengths, num_pos, nums, targets)
                self._current_batch += 1

            self._current_epoch += 1
            self._current_batch = 0


if __name__ == "__main__":

    print("Loading data...")

    MAX_SEQ_LENGTH, MAX_ARGS = get_max_length("./data/wap/")
    print("max seq length", MAX_SEQ_LENGTH)
    print("max num args", MAX_ARGS)

    train = load_corpus("./data/wap/train.txt", MAX_SEQ_LENGTH, MAX_ARGS)
    dev = load_corpus("./data/wap/dev.txt", MAX_SEQ_LENGTH, MAX_ARGS)
    test = load_corpus("./data/wap/test.txt", MAX_SEQ_LENGTH, MAX_ARGS)
    debug = load_corpus("./data/wap/debug.txt", MAX_SEQ_LENGTH, MAX_ARGS)

    print("#train", len(train.tokens))
    print("#dev", len(dev.tokens))
    print("#test", len(test.tokens))
    print("#debug", len(debug.tokens))
    print("vocab size", len(vocab))

    data = ArithmeticData(train, dev, test, debug)

    debug_batcher = ArithmeticDatasetBatcher(debug, batch_size=5)

    for _ in range(10):
        batch = debug_batcher.next_batch()
        print('--' * 50)
        print("current epoch", debug_batcher._current_epoch)
        print("current batch", debug_batcher._current_batch)
        print(batch.nums)
