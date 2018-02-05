from collections import namedtuple

import numpy as np
import tensorflow as tf

from experiments.nam_seq2seq import NAMSeq2Seq
from experiments.data import load_data, DatasetBatcher

import logging

# logging.basicConfig(level=logging.DEBUG)
import time

np.set_printoptions(linewidth=20000, precision=2, suppress=True)

SUMMARY_LOG_DIR = "./tmp/sort/summaries"


tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.app.flags.DEFINE_float("learning_rate", 1.0, "Learning rate")

tf.app.flags.DEFINE_integer("train_num_steps", -1, "Training phase - number of steps")
tf.app.flags.DEFINE_integer("train_stack_size", -1, "Training phase - stack size")

tf.app.flags.DEFINE_integer("test_num_steps", -1, "Testing phase - number of steps")
tf.app.flags.DEFINE_integer("test_stack_size", -1, "Testing phase - stack size")

tf.app.flags.DEFINE_integer("min_return_width", 5, "Minimum return width")

tf.app.flags.DEFINE_integer("eval_every", 5, "Evaluate every n-th epoch")

tf.app.flags.DEFINE_integer("max_epochs", 500, "Maximum number of epochs")

tf.app.flags.DEFINE_string("id", "x", "unique id for summary purposes")

tf.app.flags.DEFINE_float("init_weight_stddev", 1.0, "Standard deviation for initial weights")

tf.app.flags.DEFINE_float("max_grad_norm", 2.0, "Clip gradients to this norm.")

tf.app.flags.DEFINE_float("grad_noise_eta", 0.0, "Gradient noise scale.")

tf.app.flags.DEFINE_float("grad_noise_gamma", 0.55, "Gradient noise gamma.")


tf.app.flags.DEFINE_boolean("save_summary", True, "Save summary files.")


tf.app.flags.DEFINE_string("sketch", "./experiments/sort/sketch_compare.d4", "Sketch.")

tf.app.flags.DEFINE_string("data", './data/bubble/train2_test64/', "Data.")


def print_flags(flags):
    print("Flag values")
    for k, v in flags.__dict__['__flags'].items():
        print('  ', k, ':', v)

FLAGS = tf.app.flags.FLAGS


d4InitParams = namedtuple("d4InitParams",
                          "stack_size value_size batch_size min_return_width init_weight_stddev")

TrainParams = namedtuple("TrainParams",
                         "train learning_rate num_steps max_grad_norm "
                         + "grad_noise_eta grad_noise_gamma")

TestParams = namedtuple("TestParams", "stack_size num_steps")


def main(_):
    datasets = load_data(FLAGS.data)

    def load_scaffold_from_file(filename):
        with open(filename, "r") as f:
            scaffold = f.read()
        return scaffold

    sketch = load_scaffold_from_file(FLAGS.sketch)

    # calculate value_size automatically
    value_size = max(datasets.train.input_seq.max(), datasets.train.target_seq.max(),
                     datasets.dev.input_seq.max(), datasets.dev.target_seq.max(),
                     datasets.test.input_seq.max(), datasets.test.target_seq.max(),
                     datasets.debug.input_seq.max(), datasets.debug.target_seq.max()) + 2

    dataset_train = datasets.train
    dataset_dev = datasets.dev
    dataset_test = datasets.test

    train_batcher = DatasetBatcher(dataset_train, FLAGS.batch_size)

    train_seq_len = dataset_train.input_seq[:, -1].max()
    test_seq_len = dataset_test.input_seq[:, -1].max()
    dev_seq_len = dataset_dev.input_seq[:, -1].max()
    # test_seq_len = 64

    def num_steps(seq_len):
        return (seq_len * 6 + 5) * seq_len + 4

    def stack_size(seq_len):
        return seq_len * 2 + 10

    train_num_steps = num_steps(train_seq_len)
    test_num_steps = num_steps(test_seq_len)
    dev_num_steps = num_steps(dev_seq_len)

    print('train_num_steps', train_num_steps)
    print('test_num_steps', test_num_steps)

    train_stack_size = stack_size(train_seq_len)
    test_stack_size = stack_size(test_seq_len)

    print('train_stack_size', train_stack_size)
    print('test_stack_size', test_stack_size)

    FLAGS.train_num_steps = (train_num_steps if FLAGS.train_num_steps == -1
                             else FLAGS.train_num_steps)
    FLAGS.train_stack_size = (train_stack_size if FLAGS.train_stack_size == -1
                              else FLAGS.train_stack_size)

    FLAGS.test_num_steps = (test_num_steps if FLAGS.test_num_steps == -1
                            else FLAGS.test_num_steps)
    FLAGS.test_stack_size = (test_stack_size if FLAGS.test_stack_size == -1
                             else FLAGS.test_stack_size)

    print('--')
    print(' train_seq_len', train_seq_len)
    print(' test_seq_len', test_seq_len)
    print(' value_size', value_size)
    print('--')
    print_flags(FLAGS)
    print('-' * 20)

    d4_params = d4InitParams(stack_size=FLAGS.train_stack_size,
                             value_size=value_size,
                             batch_size=FLAGS.batch_size,
                             min_return_width=FLAGS.min_return_width,
                             init_weight_stddev=FLAGS.init_weight_stddev
                             )

    train_params = TrainParams(train=True,
                               learning_rate=FLAGS.learning_rate,
                               num_steps=FLAGS.train_num_steps,
                               max_grad_norm=FLAGS.max_grad_norm,
                               grad_noise_eta=FLAGS.grad_noise_eta,
                               grad_noise_gamma=FLAGS.grad_noise_gamma
                               )

    test_params = TestParams(num_steps=FLAGS.test_num_steps,
                             stack_size=FLAGS.test_stack_size
                             )

    model = NAMSeq2Seq(sketch, d4_params, train_params, test_params,
                       debug=False,
                       adjust_min_return_width=True,
                       argmax_pointers=True,
                       argmax_stacks=True,
                       )

    model.build_graph()

    # summary use can be quite slow
    use_summary = False

    best_accuracy = 0.0

    # where to save checkpoints for test set calculation
    directory_save = "./tmp/sort/checkpoints/{0}/".format(FLAGS.id)

    # directory_save needs to exist
    import os
    if not os.path.exists(directory_save):
        os.makedirs(directory_save)

    with tf.Session() as sess:

        # if True:
        #     model.load_mode(sess, directory_save + "model.checkpoint")
        #     # accuracy, partial_accuracy = model.run_eval_step(sess, dataset_dev)
        #     exit(0)

        if use_summary:
            print(' ..creating summary writer')
            summary_writer = tf.train.SummaryWriter(SUMMARY_LOG_DIR + "/" + FLAGS.id,
                                                    tf.get_default_graph())

        sess.run(tf.initialize_all_variables())

        # run max_epochs times

        print("epoch\titer\tloss\taccuracy\tpartial accuracy")

        stop_early = False
        epoch = 0
        while epoch < FLAGS.max_epochs and (not stop_early):
            epoch += 1

            total_loss = 0.0

            for i in range(train_batcher._batch_number):
                batch = train_batcher.next_batch()

                # start = time.time()
                _, loss, summaries, global_step = model.run_train_step(sess, batch, epoch)
                # end = time.time()
                # print('batch time: ', end-start)

                if use_summary:
                    summary_writer.add_summary(summaries, global_step)

                total_loss += loss

            loss_per_epoch = total_loss / (train_batcher._batch_number * train_batcher._batch_size)
            print("train\t{0}\tl:{1}".format(epoch, loss_per_epoch))

            if epoch % FLAGS.eval_every == 0:

                print('evaluating dev set for {0} steps'.format(dev_num_steps))
                accuracy, partial_accuracy = model.run_eval_step(sess, dataset_dev, dev_num_steps)

                print("dev\t{0}\ta:{1}\tpa:{2}".format(epoch, accuracy, partial_accuracy))

                if partial_accuracy > best_accuracy:
                    best_accuracy = partial_accuracy
                    print('Saving model...')
                    model.save_model(sess, directory_save + "model.checkpoint",
                                     global_step=global_step)
                    if partial_accuracy == 1.0:
                        acc, p_acc = model.run_eval_step(sess, dataset_test, test_num_steps)
                        print("test\t{0}\ta:{1}\tpa:{2}".format(epoch, acc, p_acc))
                        exit(0)

                if use_summary:
                    summary_acc = tf.Summary(value=[
                        tf.Summary.Value(tag="accuracy/accuracy",
                                         simple_value=accuracy)])
                    summary_part_acc = tf.Summary(value=[
                        tf.Summary.Value(tag="accuracy/partial_accuracy",
                                         simple_value=partial_accuracy)])

                    summary_writer.add_summary(summary_acc, global_step)
                    summary_writer.add_summary(summary_part_acc, global_step)

                    summary_writer.flush()


if __name__ == "__main__":
    tf.app.run()
