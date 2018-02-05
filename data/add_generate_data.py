import numpy as np
import os


data_directory = './data/'


class DataConfig:
    attributes = ["train", "dev", "test", "debug"]
    train = None
    dev = None
    test = None
    debug = None


# Note, these genration procedures are indicative of what I used
# In reality, I used the same train/dev/debug with different test

# TEST 8


class ConfigTrain2Test8(DataConfig):
    train = {'n': 512, 'seq_len': 1}
    dev = {'n': 256, 'seq_len': 3}
    test = {'n': 1024, 'seq_len': 4}
    debug = {'n': 10, 'seq_len': 2}


class ConfigTrain4Test8(DataConfig):
    train = {'n': 512, 'seq_len': 2}
    dev = {'n': 256, 'seq_len': 3}
    test = {'n': 1024, 'seq_len': 4}
    debug = {'n': 10, 'seq_len': 2}


class ConfigTrain8Test8(DataConfig):
    train = {'n': 512, 'seq_len': 4}
    dev = {'n': 256, 'seq_len': 3}
    test = {'n': 1024, 'seq_len': 4}
    debug = {'n': 10, 'seq_len': 2}


# TEST 64


class ConfigTrain2Test64(DataConfig):
    train = {'n': 512, 'seq_len': 1}
    dev = {'n': 256, 'seq_len': 3}
    test = {'n': 1024, 'seq_len': 32}
    debug = {'n': 10, 'seq_len': 2}


class ConfigTrain4Test64(DataConfig):
    train = {'n': 512, 'seq_len': 2}
    dev = {'n': 256, 'seq_len': 3}
    test = {'n': 1024, 'seq_len': 32}
    debug = {'n': 10, 'seq_len': 2}


class ConfigTrain8Test64(DataConfig):
    train = {'n': 512, 'seq_len': 4}
    dev = {'n': 256, 'seq_len': 3}
    test = {'n': 1024, 'seq_len': 32}
    debug = {'n': 10, 'seq_len': 2}


# TRAIN 24 TEST 128


class ConfigTrain24Test128(DataConfig):
    train = {'n': 512, 'seq_len': 12}
    dev = {'n': 256, 'seq_len': 3}
    test = {'n': 1024, 'seq_len': 64}
    debug = {'n': 10, 'seq_len': 2}


# VARYING NO OF TRAIN EXAMPLES


class ConfigVaryingTrainNo(DataConfig):
    train = {'n': None, 'seq_len': 4}
    dev = {'n': 256, 'seq_len': 16}
    test = {'n': 1024, 'seq_len': 8}
    debug = {'n': 10, 'seq_len': 2}


def _adder(xs):
    rev = xs[::-1]
    out = []
    # ln = rev[0]
    carry = rev[1]
    for i in range(2, len(rev), 2):
        sm = rev[i] + rev[i+1] + carry
        res = sm % 10
        carry = 0
        if sm >= 10:
            carry = 1
        out.append(res)
    out.append(carry)

    return out[::-1]


def generate_adder_data(num_examples, seq_len=1):
    arg_low = 0
    arg_high = 9
    digits = np.random.random_integers(arg_low, arg_high, [num_examples, seq_len * 2])
    carries = np.random.random_integers(0, 1, [num_examples, 1])
    lens = np.ones([num_examples, 1], dtype=int) * seq_len
    x = np.hstack([digits, carries, lens])
    y = np.array([_adder(item) for item in x])
    return x, y


def write_to_file(config: DataConfig, subdirectory):

    for item in config.attributes:
        dataset = getattr(config, item)
        n = dataset["n"]
        seq_len = dataset["seq_len"]

        _x, _y = generate_adder_data(n, seq_len=seq_len)
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
    write_to_file(ConfigTrain2Test8, subdirectory="add/train2_test8/")
    write_to_file(ConfigTrain4Test8, subdirectory="add/train4_test8/")
    write_to_file(ConfigTrain8Test8, subdirectory="add/train8_test8/")

    write_to_file(ConfigTrain2Test64, subdirectory="add/train2_test64/")
    # write_to_file(ConfigTrain4Test64, subdirectory="add/train4_test64/")
    # write_to_file(ConfigTrain8Test64, subdirectory="add/train8_test64/")

    for n_examples in range(2, 15):
        ConfigVaryingTrainNo.train['n'] = 2**n_examples
        write_to_file(ConfigVaryingTrainNo,
                      subdirectory="add/_train{0}/".format(2**n_examples))
