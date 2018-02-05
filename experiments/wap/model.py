import logging
import tensorflow as tf
import numpy as np
from collections import namedtuple

from d4.interpreter import SimpleInterpreter
from d4.dsm.loss import CrossEntropyLoss
from experiments.wap.load_data import load_all_datasets

from experiments.wap.load_data import ArithmeticDataset

from d4.dsm.extensible_dsm import print_dsm_state_np
# just for prettier printing...

logging.basicConfig(level=logging.ERROR)

eps = 0.00000000000000000000001

d4InitParams = namedtuple("d4InitParams", "stack_size value_size batch_size min_return_width")

DataParams = namedtuple("DataParams", "vocab_size max_length max_args")

TrainParams = namedtuple("TrainParams", "train learning_rate num_steps_train num_steps_test")


class AlgebraD4Model(object):
    def __init__(self, sketch, d4_params, data_params, train_params):

        self.sketch = sketch

        self.train = train_params.train
        self.learning_rate = train_params.learning_rate
        self.num_steps_train = train_params.num_steps_train
        self.num_steps_test = train_params.num_steps_test

        self.stack_size = d4_params.stack_size
        self.value_size = d4_params.value_size
        self.min_return_width = d4_params.min_return_width
        self.batch_size = d4_params.batch_size

        self.vocab_size = data_params.vocab_size
        self.max_length = data_params.max_length
        self.max_args = data_params.max_args

        self.vector_size = self.value_size // 2

        self.interpreter = None

        self._summaries = None
        self._DEBUG = False
        self.global_step = None
        self.saver = None

    def _add_placeholders(self):
        with tf.name_scope("input"):
            # [batch_size x max_length]
            self._pl_text = tf.placeholder(dtype=tf.int32,
                                           shape=[self.batch_size, self.max_length],
                                           name="text")
            # [batch_size]
            self._pl_text_len = tf.placeholder(dtype=tf.int64,
                                               shape=[self.batch_size],
                                               name="text_len")
            # [batch_size x max_args]
            self._pl_num_pos = tf.placeholder(dtype=tf.int32,
                                              shape=[self.batch_size, self.max_args],
                                              name="num_pos")
            # [batch_size x max_args]
            self._pl_nums = tf.placeholder(dtype=tf.float32,
                                           shape=[self.batch_size, self.max_args],
                                           name="nums")

    def _add_birnn(self):
        # TODO word2vec / GLoVe
        # [vocab_size, vector_size]
        with tf.name_scope("embedding"):
            embeddings = tf.Variable(tf.random_normal([self.vocab_size, self.vector_size],
                                                      mean=0.0, stddev=0.1, dtype=tf.float32),
                                     name="embedding_matrix")

            # [batch_size x max_length x vector_size]
            input_embedded = tf.nn.embedding_lookup(embeddings,
                                                    self._pl_text,
                                                    name="embedding_lookup")

        inputs = input_embedded
        with tf.name_scope("BiRNN"):
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.vector_size, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.vector_size, state_is_tuple=True)
            # L[max_seq_length] of [batch_size x 2 vector_size]
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                         dtype=tf.float32,
                                                         sequence_length=self._pl_text_len)
            outputs = tf.concat(2, outputs)

        with tf.name_scope("output"):
            outputs = tf.unpack(outputs)
            with tf.name_scope("argument_embeddings"):
                num_positions = tf.unpack(self._pl_num_pos)
                # L[batch_size] of [max_args x 2 vector_size]
                arg_embeddings = [tf.gather(outputs[i], num_positions[i])
                                  for i in range(self.batch_size)]
                # [batch_size x max_args x 2 vector_size]
                self._arg_emb = tf.pack(arg_embeddings)

            with tf.name_scope("question_embedding"):
                # last output vectors (per direction) concatenated into a question_embedding
                question_embedding = []
                for i in range(self.batch_size):
                    bi_start = tf.gather(outputs[i], 0)
                    bi_end = tf.gather(outputs[i], self._pl_text_len[i] - 1)
                    end1 = tf.slice(bi_end, [0], [10])
                    end2 = tf.slice(bi_start, [10], [-1])
                    q = tf.concat(0, [end1, end2])
                    question_embedding.append(q)
                # [batch_size x 2 vector_size]
                self._q_emb = tf.pack(question_embedding)

    def _assemble_heap(self):
        with tf.name_scope("heap_assembly"):
            with tf.name_scope("numbers"):
                # [2 vector_size x 1 x batch_size]
                # TODO one-hot

                # [max_arg, batch_size]

                numbers = tf.transpose(
                    tf.one_hot(tf.cast(self._pl_nums, dtype=tf.int32), self.value_size),
                    perm=[2, 1, 0])

                # numbers = tf.transpose(self._pl_nums, perm=[1, 0])
                # print('---', numbers)
                # numbers = tf.expand_dims(numbers, 0)
                # numbers = tf.pad(numbers, [[0, self.value_size - 1], [0, 0], [0, 0]], "CONSTANT")
                # print('---', numbers)
            # [2 vector_size x 1 x batch_size]
            empty = tf.zeros([self.value_size, 1, self.batch_size], name="empty_values")
            with tf.name_scope("question_embedding"):
                # [2 vector_size x 1 x batch_size]
                q = tf.expand_dims(tf.transpose(self._q_emb, perm=[1, 0]), 1)
            with tf.name_scope("argument_embeddings"):
                # [2 vector_size x max_args, batch_size]
                arg = tf.transpose(self._arg_emb, perm=[2, 1, 0])
            with tf.name_scope("heap"):
                # [2 vector_size x heap_size x batch_size]
                self._init_heap = tf.concat(
                    1, [empty, q, arg, empty, numbers] + [empty] * (self.value_size - 9))

    def _add_nam(self):
        self.interpreter = SimpleInterpreter(stack_size=self.stack_size,
                                             value_size=self.value_size,
                                             min_return_width=self.min_return_width,
                                             batch_size=self.batch_size)
        for batch in range(self.batch_size):
            self.interpreter.load_code(self.sketch, batch)

        if self.train:
            self.interpreter.set_heap(self._init_heap)

        self.interpreter.create_initial_dsm()

        trace = self.interpreter.execute(self.num_steps_train)
        # self._trace = trace
        # self._dsm_loss = L2Loss(trace[-1], self.interpreter)
        self._dsm_loss = CrossEntropyLoss(trace[-1], self.interpreter)
        self._loss = self._dsm_loss.loss
        tf.scalar_summary('loss_per_batch', tf.minimum(1000.0, self._loss))

    def _add_train(self):
        with tf.name_scope("optimiser"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            vars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(self._loss, vars)
            # grads_and_vars = self._grad_add_noise(grads_and_vars, 0.1)
            grads_and_vars = self._grad_clip_by_norm(grads_and_vars, 1.0)

            self._train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    # dirty tricks....done dirt cheap
    # oh!

    @staticmethod
    def _grad_clip_by_norm(grads_and_vars, norm):
        with tf.name_scope("grad_clip_by_norm"):
            grads, vars = zip(*grads_and_vars)
            clipped_grads, _ = tf.clip_by_global_norm(grads, norm)
            return list(zip(clipped_grads, vars))

    @staticmethod
    def _grad_add_noise(grads_and_vars, scale):
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
    def _entropy(input):
        with tf.name_scope("entropy"):
            input += eps
            entropy = input * tf.log(input)
            return - tf.reduce_sum(entropy)

    @staticmethod
    def _limit_log(input):
        with tf.name_scope("limit_logits"):
            return tf.log(tf.maximum(input, eps))

    @staticmethod
    def _normalise(input, stop_gradient=True):
        with tf.name_scope("normalise"):
            norm = input / tf.reduce_sum(tf.abs(input), 0, keep_dims=True)
            if stop_gradient:
                return tf.stop_gradient(norm)
            else:
                return norm

    def build_graph(self):
        print('Building graph')
        self._add_placeholders()
        self._add_birnn()
        self._assemble_heap()
        self._add_nam()
        self.saver = tf.train.Saver()
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if self.train:
            self._add_train()
        print('Building complete')

        self._summaries = tf.merge_all_summaries()

    def _build_birnn_feed(self, data_batch):
        feed = {
            self._pl_text: data_batch.tokens,
            self._pl_text_len: data_batch.seq_lengths,
            self._pl_num_pos: data_batch.num_pos,
            self._pl_nums: data_batch.nums
        }
        return feed

    def _complete_feed(self, data_batch):
        # BiRNN input feed
        feed = self._build_birnn_feed(data_batch)
        # total input feed
        return self._dsm_loss.current_feed_dict(feed)

    def run_train_step(self, sess, data_batch: ArithmeticDataset):
        feed_in = self._complete_feed(data_batch)

        # desired output
        feed_out = [self._train_op, self._summaries, self._loss, self.global_step]

        for j in range(self.batch_size):
            # self.interpreter.load_stack(data_batch.input_seq[j], j)
            # print(data_batch.targets[j])
            self._dsm_loss.load_target_stack(data_batch.targets[j], j)

        return sess.run(feed_out, feed_dict=feed_in)

    def run_eval_step(self, sess, data: ArithmeticDataset, debug=False):
        # BiRNN input feed

        data_tokens = data.tokens
        data_seq_lengths = data.seq_lengths
        data_num_pos = data.num_pos
        data_nums = data.nums
        data_targets = data.targets

        num_batches = len(data_targets) // self.batch_size
        rest = len(data_targets) % self.batch_size

        dataset_size = len(data_targets)

        if rest != 0:
            num_batches += 1

            pad = lambda x: np.pad(x, ((0, (self.batch_size - rest)), (0, 0)), 'edge')

            data_tokens = pad(data_tokens)
            data_seq_lengths = np.pad(data_seq_lengths, (0, self.batch_size - rest), 'edge')
            data_num_pos = pad(data_num_pos)
            data_nums = pad(data_nums)
            data_targets = pad(data_targets)

        counter = 0

        hits = 0.0
        for i in range(num_batches):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size

            data_batch = ArithmeticDataset(data_tokens[start:end],
                                           data_seq_lengths[start:end],
                                           data_num_pos[start:end],
                                           data_nums[start:end],
                                           data_targets[start:end])

            # print(data_batch.targets)
            feed = self._build_birnn_feed(data_batch)

            # evaluate the constructed (fully batched) heap and load it into the interpreter
            heap = self._init_heap.eval(feed, sess)
            self.interpreter.load_heap(heap, batch=None)
            # print(heap[:, :, 0])

            # execute the machine
            trace, _ = self.interpreter.execute_test_time(sess, self.num_steps_test,
                                                          external_feed_dict=feed,)
                                                        # use_argmax_pointers=True,
                                                        # use_argmax_stacks=True,
                                                        # data=(data_nums[start:end],
                                                        #       data_targets[start:end]))

            last_state = trace[-1]

            DS = last_state[self.interpreter.test_time_data_stack]
            DSP = last_state[self.interpreter.test_time_data_stack_pointer]

            if (debug):
                for j in range(self.num_steps_test):
                    print('---------', j)
                    ds = trace[j][self.interpreter.test_time_data_stack]
                    dp = trace[j][self.interpreter.test_time_data_stack_pointer]
                    return_stack = trace[j][self.interpreter.test_time_return_stack]
                    return_stack_pointer = trace[j][self.interpreter.test_time_return_stack_pointer]
                    pc = trace[j][self.interpreter.test_time_pc]
                    heap = trace[j][self.interpreter.test_time_heap]

                    print_dsm_state_np(ds, dp,
                                       # return_stack=return_stack,
                                       # return_stack_pointer=return_stack_pointer,
                                       pc=pc,
                                       interpreter=self.interpreter)

            to = rest if i == num_batches - 1 and rest != 0 else self.batch_size

            for j in range(to):
                counter += 1
                dstack_p = np.argmax(DSP[:, j])
                # print(np.argmax(DS[:, :, i], 0))
                result = np.argmax(DS[:, :, j], 0)[dstack_p: dstack_p + 1]
                # print(data_batch.targets[i], 'vs', result)
                # print(data_batch.targets[j], result)
                if data_batch.targets[j] == result:
                    hits += 1.0

        accuracy = hits / dataset_size

        print('      correct hits:', hits)
        print('      instances:', dataset_size)
        print('      counter:', counter)
        print('    accuracy:', accuracy)

        return accuracy

        # print('---> EVAL')
        #
        # print('data_batch.targets', data_batch.targets)
        #
        # for i in range(1, len(trace)):
        #     state = trace[i]
        #     DS = state[self.interpreter.test_time_data_stack]
        #     DSP = state[self.interpreter.test_time_data_stack_pointer]
        #
        #     print('DS', DS[:, :, 0])
        #     print('DSP', DSP[:, 0])
        #
        #     feed_dict[self.test_time_data_stack] = sharpen_stck(data_stack)
        #     feed_dict[self.test_time_data_stack_pointer] = sharpen(data_stack_pointer)
        #
        #     feed_dict[self.test_time_return_stack] = sharpen_stck(return_stack)
        #     feed_dict[self.test_time_return_stack_pointer] = sharpen(return_stack_pointer)
        #
        #     # TODO: sharpen heap
        #     feed_dict[self.test_time_heap] = heap
        #
        #     feed_dict[self.test_time_pc] = sharpen(pc)

    def run_test_step(self, sess, data_batch):
        # BiRNN input feed
        feed = self._build_birnn_feed(data_batch)

        # evaluate the constructed heap and load it into the interpreter
        heap = self._init_heap.eval(feed, sess)
        self.interpreter.load_heap(heap, batch=None)

        # execute the machine
        trace, _ = self.interpreter.execute_test_time(sess, self.num_steps_test,
                                                      external_feed_dict=feed)

        print(">>>>>>>>>> DEBUGIIIING !!!! >>>>>>>>>>")
        # print(data_batch.targets)

        print('init')
        # print(heap[:, :, 0])

        for i in range(1, len(trace)):
            print('---', i, '---')

            ds = trace[i][self.interpreter.test_time_data_stack]
            dp = trace[i][self.interpreter.test_time_data_stack_pointer]
            return_stack = trace[i][self.interpreter.test_time_return_stack]
            return_stack_pointer = trace[i][self.interpreter.test_time_return_stack_pointer]
            pc = trace[i][self.interpreter.test_time_pc]
            heap = trace[i][self.interpreter.test_time_heap]

            print_dsm_state_np(ds, dp,
                               return_stack=return_stack,
                               return_stack_pointer=return_stack_pointer,
                               pc=pc, heap=heap,
                               interpreter=self.interpreter)

            # if i > 1:
            #
            #     # for n in tf.get_default_graph().as_graph_def().node:
            #     #     if not 'optimiser' in n.name:
            #     #         print(n.name)
            #     # print(tf.all_variables())
            #
            #     t = tf.get_default_graph().get_tensor_by_name(
            #         ("test/execute/encoder_decoder/decoder/"
            #          "permute_decoder/permute_attention/Softmax:0"))
            #     out = sess.run(t, feed_dict=trace[i])
            #     print('permute attention')
            #     print(out)
            #
            #     # t = tf.get_default_graph().get_tensor_by_name(
            #     #     ("train_step_{0}/execute/encoder_decoder_1/decoder"
            #     #      "/choice_decoder/choice_attention/Softmax:0".format(i-1)))
            #     tx = tf.get_default_graph().get_tensor_by_name(
            #         ("test/execute/encoder_decoder_1/decoder"
            #          "/choice_decoder/choice_attention/transpose:0".format(i-1)))
            #     out = sess.run(tx, feed_dict=trace[i])
            #     print('choice1 attention, unnormalised')
            #     print(out)
            #
            #     t = tf.get_default_graph().get_tensor_by_name(
            #         ("test/execute/encoder_decoder_1/decoder/"
            #          "choice_decoder/choice_attention/Softmax:0".format(i-1)))
            #     out = sess.run(t, feed_dict=trace[i])
            #     print('choice1 attention')
            #     print(out)
            #
            #     # t = tf.get_default_graph().get_tensor_by_name(
            #     #     ("train_step_{0}/execute/encoder_decoder_2/decoder/"
            #     #      "choice_decoder/choice_attention/Softmax:0".format(i-1)))
            #     t = tf.get_default_graph().get_tensor_by_name(
            #         ("test/execute/encoder_decoder_2/decoder/"
            #          "choice_decoder/choice_attention/Softmax:0".format(i-1)))
            #     out = sess.run(t, feed_dict=trace[i])
            #     print('choice2 attention')
            #     print(out)

            if i == 1:
                print_dsm_state_np(ds, dp,
                                   return_stack=return_stack,
                                   return_stack_pointer=return_stack_pointer,
                                   pc=pc, heap=heap,
                                   interpreter=self.interpreter)
            else:
                print_dsm_state_np(ds, dp,
                                   return_stack=return_stack,
                                   return_stack_pointer=return_stack_pointer,
                                   pc=pc,
                                   interpreter=self.interpreter)

        # t1 = tf.get_default_graph().get_tensor_by_name(
        #     "train_step_1/execute/merge_dsms_2/data_stack/data_stack/Sum:0")
        # t2 = tf.get_default_graph().get_tensor_by_name(
        #     "train_step_1/execute/merge_dsms_2/data_stack/data_stack/Sum:0")
        # t3 = tf.get_default_graph().get_tensor_by_name(
        #     "train_step_2/execute/choice_decoder/WRITE_BUFFER/write_buffer/add:0")
        # t4 = tf.get_default_graph().get_tensor_by_name(
        #     "train_step_2/execute/choice_decoder/WRITE_BUFFER_1/write_buffer/add:0")
        # t5 = tf.get_default_graph().get_tensor_by_name(
        #     "train_step_2/execute/choice_decoder/WRITE_BUFFER_2/write_buffer/add:0")
        #
        # print('merge_dsms')
        # o1, o2, o3, o4, o5 = sess.run([t1, t2, t3, t4, t5], feed_dict=feed_in)

        return [trace]

    def save_model(self, sess, name, global_step=None):
        print('  ..saving model..')
        self.saver.save(sess, name, global_step=global_step)
        print('  ..model saved')

    def load_model(self, sess, directory):
        print('  ..loading model')
        latest = tf.train.latest_checkpoint(directory)
        self.saver.restore(sess, latest)
        print('  ..model loaded')

if __name__ == "__main__":
    # dataset load
    data, MAX_LENGTH, MAX_ARGS, VOCAB_SIZE = load_all_datasets()

    # parameter setup
    MAX_LENGTH += 2
    BATCH_SIZE = 5
    VALUE_SIZE = 20
    MIN_RETURN_WIDTH = VALUE_SIZE
    STACK_SIZE = 10

    d4_params = d4InitParams(stack_size=STACK_SIZE,
                             value_size=VALUE_SIZE,
                             batch_size=BATCH_SIZE,
                             min_return_width=MIN_RETURN_WIDTH)

    data_params = DataParams(vocab_size=VOCAB_SIZE,
                             max_length=MAX_LENGTH,
                             max_args=MAX_ARGS)

    train_params = TrainParams(train=True,
                               learning_rate=0.01,
                               num_steps_train=6,
                               num_steps_test=10)

    def load_sketch_from_file(filename):
        with open(filename, "r") as f:
            scaffold = f.read()
        return scaffold

    sketch = load_sketch_from_file("./experiments/wap/algebra_sketch.d4")

    print(sketch)

    model = AlgebraD4Model(sketch, d4_params, data_params, train_params)

    model.build_graph()
