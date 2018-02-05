from collections import namedtuple

import numpy as np
import tensorflow as tf

from experiments.data import SeqDataset
from d4.dsm.loss import CrossEntropyLoss
from d4.interpreter import SimpleInterpreter

# logging.basicConfig(level=logging.DEBUG)


np.set_printoptions(linewidth=20000, precision=2, suppress=True)


d4InitParams = namedtuple(
    "d4InitParams", "stack_size value_size batch_size min_return_width init_weight_stddev"
)

TrainParams = namedtuple(
    "TrainParams", "train learning_rate num_steps max_grad_norm grad_noise_eta grad_noise_gamma"
)

TestParams = namedtuple("TestParams", "stack_size num_steps")


class NAMSeq2Seq:
    def __init__(self,
                 sketch,
                 d4_params: d4InitParams,
                 train_params: TrainParams,
                 test_params: TestParams,
                 debug=False,
                 adjust_min_return_width=True,
                 use_slot_l2_regularizer=False,
                 argmax_pointers=False,
                 argmax_stacks=False
                 ):

        self.sketch = sketch

        self.stack_size = d4_params.stack_size
        self.value_size = d4_params.value_size
        self.batch_size = d4_params.batch_size
        self.min_return_width = d4_params.min_return_width
        self.init_weight_stddev = d4_params.init_weight_stddev

        # train params
        self.train_num_steps = train_params.num_steps
        self.learning_rate = train_params.learning_rate
        self.max_grad_norm = train_params.max_grad_norm

        self.grad_noise_eta = train_params.grad_noise_eta
        self.grad_noise_gamma = train_params.grad_noise_gamma

        # test params
        if test_params.stack_size > self.stack_size:
            self.stack_size = test_params.stack_size

        self.test_time_num_steps = test_params.num_steps

        self.saver = None

        # TODO the rest

        self.argmax_pointers = argmax_pointers
        self.argmax_stacks = argmax_stacks

        self.debug = debug
        self.adjust_min_return_width = adjust_min_return_width
        if self.adjust_min_return_width:
            self.min_return_width += self.stack_size

        # TODO isolate the l2 regulariser parameter
        self.use_slot_l2_regularizer = use_slot_l2_regularizer

        # internal
        self._summaries = None

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.epoch = tf.Variable(0, name="epoch", trainable=False, dtype=tf.float32)

        # usually eta / (1+t)^gamma
        self._grad_noise_scale = lambda epoch: (tf.pow(1 + epoch, -self.grad_noise_gamma)
                                                * self.grad_noise_eta)

        # self._epoch = 0

    def _add_nam(self):
        self.interpreter = SimpleInterpreter(stack_size=self.stack_size,
                                             value_size=self.value_size,
                                             min_return_width=self.min_return_width,
                                             batch_size=self.batch_size,
                                             init_weight_stddev=self.init_weight_stddev
                                             )

        if self.debug:
            print(" ..loading code")
        for batch in range(self.batch_size):
            self.interpreter.load_code(self.sketch, batch)
        self.interpreter.create_initial_dsm()

        if self.debug:
            print(" ..executing interpreter")
        trace = self.interpreter.execute(self.train_num_steps)
        self._trace = trace

        if self.debug:
            print(" ..building loss")
        self._dsm_loss = CrossEntropyLoss(trace[-1], self.interpreter)

        l2_loss = self._dsm_loss.loss

        # if self.use_slot_l2_regularizer:
        #     l2_loss = self._dsm_loss.regularised_l2_loss

        # if self.penalise_non_empty_RP:
        #     l2_loss = l2_loss + self._dsm_loss.return_stack_loss

        self._loss = l2_loss

        tf.scalar_summary('loss/train', tf.minimum(tf.constant(1000.0), self._loss))

    def _add_train(self):
        if self.debug:
            print(" ..adding training operators")
        with tf.name_scope("optimiser"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            vars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(self._loss, vars)

            grads_and_vars = self.grad_add_noise(grads_and_vars, self._grad_noise_scale(self.epoch))
            grads_and_vars = self.grad_clip_by_norm(grads_and_vars, self.max_grad_norm)
            # print(grads_and_vars)
            self._grads_and_vars = grads_and_vars
            self._train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    @staticmethod
    def grad_add_noise(grads_and_vars, scale):
        with tf.name_scope("grad_add_noise"):
            grads, vars = zip(*grads_and_vars)
            noisy_grads = []
            for grad in grads:
                if isinstance(grad, tf.Tensor):
                    noisy_grads.append(grad + tf.truncated_normal(tf.shape(grad)) * scale)
                else:
                    noisy_grads.append(grad)
            return list(zip(noisy_grads, vars))

    @staticmethod
    def grad_clip_by_norm(grads_and_vars, norm):
        with tf.name_scope("grad_clip_by_norm"):
            grads, vars = zip(*grads_and_vars)
            clipped_grads, _ = tf.clip_by_global_norm(grads, norm)
            return list(zip(clipped_grads, vars))

    def save_model(self, sess, name, global_step=None):
        print('  ..saving model..')
        self.saver.save(sess, name, global_step=global_step)
        print('  ..model saved')

    def load_model(self, sess, directory):
        print('  ..loading model')
        latest = tf.train.latest_checkpoint(directory)
        self.saver.restore(sess, latest)
        print('  ..model loaded')

    def build_graph(self):
        if self.debug:
            print("Building graph...")
        self._add_nam()
        # if self.train:
        self._add_train()
        self.saver = tf.train.Saver()
        self._summaries = tf.merge_all_summaries()
        if self.debug:
            print('Building complete')

    def run_train_step(self, sess, data_batch: SeqDataset, epoch):

        self.epoch.assign(epoch)

        # feeds
        feed_in = self._dsm_loss.current_feed_dict()
        feed_out = [self._train_op, self._loss, self._summaries, self.global_step]

        # feed the stack and the target stack
        for j in range(self.batch_size):
            self.interpreter.load_stack(data_batch.input_seq[j], j, last_float=False)
            self._dsm_loss.load_target_stack(data_batch.target_seq[j], j)

        return sess.run(feed_out, feed_dict=feed_in)

    def run_eval_step(self, sess, dataset: SeqDataset, max_steps):

        correct = 0
        total = 0
        partial_correct = 0
        partial_total = 0

        num_batches, superfluous_batch = divmod(len(dataset.target_seq), self.batch_size)

        if superfluous_batch > 0:
            num_batches += 1

        current_batch_size = self.batch_size

        if self.debug:
            print('Misclassified instances:')

        for batch_no in range(num_batches):

            if batch_no == num_batches - 1 and superfluous_batch > 0:
                current_batch_size = superfluous_batch

            # set up the input/target
            for batch_elem in range(current_batch_size):
                index = batch_no * self.batch_size + batch_elem
                # print(dataset.input_seq[index])
                # print(dataset.target_seq[index])

                self.interpreter.test_time_load_stack(dataset.input_seq[index], batch_elem)
                self._dsm_loss.load_target_stack(dataset.target_seq[index], batch_elem)

            # execute da thing
            test_trace, _ = self.interpreter.execute_test_time(
                sess, max_steps, use_argmax_pointers=self.argmax_pointers,
                use_argmax_stacks=self.argmax_stacks, debug=False, save_only_last_step=True)

            # pull out stacks
            final_state = test_trace[-1]
            data_stacks = final_state[self.interpreter.test_time_data_stack]
            data_stack_pointers = final_state[self.interpreter.test_time_data_stack_pointer]

            for batch_elem in range(current_batch_size):
                index = batch_no * self.batch_size + batch_elem
                # argmax everything !!
                pointer = np.argmax(data_stack_pointers[:, batch_elem])

                # print(data_stacks[:, :, 0])
                stack_output = data_stacks[:, 0:pointer+1, batch_elem]
                result = np.argmax(stack_output, 0)

                if self.debug and list(dataset.target_seq[index]) != list(result):
                    print(' in: ', dataset.input_seq[index],
                          ' expected: ', dataset.target_seq[index],
                          ' got:', result)

                # get the overlap stats
                if len(result) == len(dataset.target_seq[index]):
                    overlap = np.sum(result == dataset.target_seq[index])
                    if overlap == len(result):
                        correct += 1
                    total += 1
                    partial_correct += overlap
                    partial_total += len(result)

        # print("correct", correct)
        # print("total", total)
        #
        # print("partial_correct", partial_correct)
        # print("partial_total", partial_total)

        accuracy = correct / total if total > 0 else 0
        partial_accuracy = partial_correct / partial_total if partial_total > 0 else 0

        return accuracy, partial_accuracy
