import numpy as np
import os

data_directory = './data/sort/'


class DataConfig:
    attributes = ["train", "dev", "test", "debug"]
    train = None
    dev = None
    test = None
    debug = None


class Train2Test64(DataConfig):
    train = {'n': 256, 'seq_len': 2}
    dev = {'n': 32, 'seq_len': 8}
    test = {'n': 32, 'seq_len': 8}
    debug = {'n': 16, 'seq_len': 2}


class ConfigVaryingTrainNo(DataConfig):
    train = {'n': None, 'seq_len': 3}
    dev = {'n': 32, 'seq_len': 3}
    test = {'n': 1024, 'seq_len': 8}
    debug = {'n': 16, 'seq_len': 2}


def _sort(xs):
    ys = list(xs)
    list.sort(ys)
    return ys[::-1]


def generate(num_examples, seq_len):
    arg_low = 0
    arg_high = 9
    digits = np.random.random_integers(arg_low, arg_high, [num_examples, seq_len])
    lens = np.ones([num_examples, 1], dtype=int) * seq_len
    x = np.hstack([digits, lens])

    y = np.array([_sort(item[:-1]) for item in x])
    return x, y


def write_to_file(config: DataConfig, subdirectory):

    for item in config.attributes:
        dataset = getattr(config, item)
        n = dataset["n"]
        seq_len = dataset["seq_len"]

        _x, _y = generate(n, seq_len)

        filename = item + '.txt'

        path = data_directory + subdirectory
        full_path = path + filename

        if not os.path.exists(path):
            os.makedirs(path)

        with open(full_path, 'w') as f:
            f.write("# y\tx\n")
            for x, y in zip(_x, _y):
                f.write("[{0}]\t[{1}]\n".format(" ".join(map(str, y)),  " ".join(map(str, x))))


if __name__ == "__main__":
    write_to_file(Train2Test64, 'train2_test64/')

    for n_examples in range(2, 11):
        ConfigVaryingTrainNo.train['n'] = 2**n_examples
        write_to_file(ConfigVaryingTrainNo,
                      subdirectory="_train{0}/".format(2**n_examples))
