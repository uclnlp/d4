import sys
import abc
import copy
import tensorflow as tf
import numpy as np
from collections import defaultdict

import d4.intermediate as im
import d4.dsm.symbolic_dsm as sym


def symbol_match(sym1, sym2):
    """
    Check whether two symbols match. If one argument is None they always match.
    :param sym1: symbol 1
    :param sym2: symbol 2
    :return: whether both symbol (sequences) match.
    """
    if len(sym1) != len(sym2):
        return False
    for e1, e2 in zip(sym1, sym2):
        if not (e1 == e2 or e1 is None or e2 is None):
            return False
    return True


class AssembledWord(metaclass=abc.ABCMeta):
    """
    An assembled word is a function from DSM state to DSM state.
    """

    def __init__(self):
        # batch -> integer list of locations of this word in code.
        self.indices_in_batches = defaultdict(list)

    def append_code_position(self, batch, position):
        self.indices_in_batches[batch].append(position)

    @abc.abstractmethod
    def __call__(self, dsm):
        pass


class PipelineWord(AssembledWord):
    """
    Pipeline of assembled words, applied iteratively to a DSM state
    """
    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def append_code_position(self, batch, position):
        for step in self.steps:
            step.append_code_position(batch, position)

    def __call__(self, dsm):
        result = dsm
        for step in self.steps:
            result = step(result)
        return result

    def __str__(self):
        return "Pipeline {}".format("\n  -> ".join([str(s) for s in self.steps]))


class ParallelWord(AssembledWord):
    """
    Parallel assembled word - used for branching
    """
    def __init__(self, true_branch: AssembledWord, false_branch: AssembledWord):
        super().__init__()
        self.true_branch = true_branch
        self.false_branch = false_branch

    def append_code_position(self, batch, position):
        self.false_branch.append_code_position(batch, position)
        self.true_branch.append_code_position(batch, position)

    def __call__(self, dsm):
        # evaluate the branching test
        test = dsm.evaluate(sym.data_stack_elem(0))
        dropped = dsm.copy()
        # dropped.data_stack_pointer = dropped.evaluate(sym.inc(sym.data_stack_pointer(),
        #                                                       2, dsm.stack_size))

        false_dsm = self.false_branch(dropped)
        true_dsm = self.true_branch(dropped)
        neg_prop = test[0:1, :]  # [1, batch_size]
        weights = tf.concat(0, [1.0 - neg_prop, neg_prop])  # [2, batch_size]
        # merge true and false branches
        merged = merge_dsms(dropped, [true_dsm, false_dsm], weights)
        return merged

    def __str__(self):
        return "PARALLEL \n   {}\n   {})".format(self.true_branch, self.false_branch)


class Halt(AssembledWord):
    """
    Stops the machine by yielding the input state as output state.
    """

    def __init__(self):
        super().__init__()
        self.name = im.halt()[0]

    def __call__(self, dsm):
        return dsm


class AssembledVocab:
    """
    Vocabulary built from assembled words
    """
    def __init__(self):
        self.words = []
        self.listeners = []

    def __len__(self):
        return len(self.words)

    def add_word(self, word: AssembledWord):
        self.words.append(word)


def merge_tensors(weights, buffers, perm, reduction_index, name="merge_tensors"):
    """
    Linear combination of a list of 3D tensor buffers.

    :param weights: [num_weights, batch_size]
    :param buffers: list of [*, *, batch_size] tensors
    :param perm: permutation to use after stacking and before merging/summing.
    :param reduction_index: index over which to reduce
    :param name: name for name_space
    :return: [*, *, batch_size] tensor that is the linear combination of the input buffers.
    """
    with tf.name_scope(name):
        stacked = tf.pack(buffers)  # [vocab_size, value_size, stack_size, batch_size]
        transposed = tf.transpose(stacked, perm)  # [value_size, stack_size, batch_size, vocab_size]
        merged = tf.reduce_sum(transposed * weights, reduction_index)
    return merged


def merge_dsms(original, dsms, word_weights, do_normalise=True):
    """
    Merge a list of DSMs using some weights. Use an original DSM as prototype.

    :param original: original DSM to copy and change (for preserving persistent attributes)
    :param dsms: the DSMs
    :param word_weights: [num_weights, batch_size]
    :param do_normalise: should pointers be normalised
    :return: sum of weighted DSMSs. Will keep persistent attributes of original DSM.
    """

    with tf.name_scope("merge_dsms"):

        def normalise(pointer):
            """ Used for normalising pointers """
            return normalise_pointer(pointer) if do_normalise else pointer

        result = original.copy()
        buffer_perm = [1, 2, 0, 3]
        value_perm = [1, 0, 2]
        # dstack

        with tf.name_scope("data_stack"):
            result.data_stack = merge_tensors(word_weights, [s.data_stack for s in dsms],
                                              buffer_perm, 2, name="data_stack")
            result.data_stack_pointer = normalise(
                merge_tensors(word_weights, [s.data_stack_pointer for s in dsms],
                              value_perm, 1, name="data_stack_pointer"))
        # rstack
        with tf.name_scope("return_stack"):
            result.return_stack = merge_tensors(word_weights, [s.return_stack for s in dsms],
                                                buffer_perm, 2, name="return_stack")
            result.return_stack_pointer = normalise(
                merge_tensors(word_weights, [s.return_stack_pointer for s in dsms],
                              value_perm, 1, name="return_stack_pointer"))
        # heap
        with tf.name_scope("heap"):
            result.heap = merge_tensors(word_weights, [s.heap for s in dsms],
                                        buffer_perm, 2, name="heap")
        # pc
        with tf.name_scope("pc"):
            result.pc = normalise(merge_tensors(word_weights, [s.pc for s in dsms],
                                                value_perm, 1, name="pc"))
        return result


def create_buffer(length, width, batch_size, value_at_zero=0.0, as_tf=True):
    """
    Creates a [width, length, batch_size] buffer with value `value_at_zero` at each (0,*,*).

    :param as_tf: should the output be transformed to a TF constant.
    :param length: length of buffer.
    :param width: width of buffer.
    :param batch_size: batch_size.
    :param value_at_zero: value to put at (0,*,*)
    :return: mostly zero buffer with values at  (0,*,*).
    """
    result = np.zeros((width, length, batch_size))
    for batch in range(0, batch_size):
        for row in range(0, length):
            result[0, row, batch] = value_at_zero
    # result[0, :, :] = value_at_zero
    return tf.constant(result, dtype=tf.float32) if as_tf else result


def create_diag_buffer(length, width, batch_size):
    """
    Creates a [width, length, batch_size] tensor where each (i,i,*) is 1.0

    :param length: length of buffer.
    :param width: width of buffer.
    :param batch_size: batch_size.
    :return: mostly zero buffer with 1 at (i,i,*).
    """
    result = np.zeros((width, length, batch_size))
    for batch in range(0, batch_size):
        for row in range(0, min(length, width)):
            result[row, row, batch] = 1.0
    # eye = np.eye(width, length)
    # result = np.repeat(eye[:, : , np.newaxis], batch_size, axis=2)
    return tf.constant(result, dtype=tf.float32)


def create_pointer(length, batch_size, index, value_at_index=1.0, as_tf=True):
    """
    Create a pointer into a buffer

    :param length: length of the buffer.
    :param batch_size: buffer size.
    :param index: index of address
    :param value_at_index: value at index.
    :param as_tf: should the result be returned as a TF constant.
    :return: batch of one hot vectors
    """
    result = np.zeros((length, batch_size))
    for batch in range(0, batch_size):
        result[index, batch] = value_at_index
    # result[index, :] = value_at_index
    return tf.constant(result, dtype=tf.float32) if as_tf else result


def normalise_pointer(pointer):
    """
    Normalises a pointer by dividing by the sum of its values

    :param pointer: matrix [width, batch_size]
    :return: pointer matrix [width, batch_size] where rows are normalised.
    """
    with tf.name_scope("normalise_pointer"):
        norm = tf.reduce_sum(tf.abs(pointer), 0, keep_dims=True)  # [1, batch_size]
        return pointer / norm


def create_inc_matrix(size, amount):
    """
    Creates a increment/decrement transition matrix.

    :param size: size of matrix.
    :param amount: amount of increment.
    :return: a matrix that when multipled with one hot vectors moves the hot element by `amount`.
    """
    result = np.zeros((size, size))
    for i in range(0, size):
        result[(i + amount) % size, i] = 1.0
    # eye = np.eye(size)
    # result = np.roll(eye, -amount, axis=1)
    return tf.constant(result, dtype=tf.float32)


def create_alg_op_matrix(size, op):
    assert op in {'add', 'sub', 'mul', 'div'}
    ret = np.zeros([size, size, size])
    for i in range(0, size):
        for j in range(0, size):
            if op == 'add':
                ret[i, j, (i + j) % size] = 1.0
            elif op == 'sub':
                ret[i, j, (i - j) % size] = 1.0
            elif op == 'mul':
                ret[i, j, (i * j) % size] = 1.0
            elif op == 'div':
                if j > 0:
                    ret[i, j, (i // j) % size] = 1.0
                elif j == 0:
                    ret[i, j, size - 1] = 1.0

    return tf.constant(ret, dtype=tf.float32)



def create_pc_inc_matrix_pair(batch_to_positions, code_size, batch_size):
    """
    Creates two matrices that can be used to increment a PC on the positions given.

    :param batch_to_positions: pairs (batch, supported positions)
    :param code_size: number of words in code.
    :param batch_size: guess :)
    :return: input_matrix, output_matrix
    """
    in_tensor = np.zeros((code_size, code_size, batch_size))
    for position in range(0, code_size):
        in_tensor[position, position, :] = 1.0
    # eye = np.eye(code_size, code_size)
    # in_tensor = np.repeat(eye[:, : , np.newaxis], batch_size, axis=2)

    out_tensor = np.zeros((code_size, code_size, batch_size))
    for position in range(0, code_size):
        out_tensor[position, position, :] = 1.0
    # out_tensor = np.copy(in_tensor)

    for batch, positions in batch_to_positions:
        for position in positions:
            out_tensor[position, position, batch] = 0.0
            out_tensor[(position + 1) % code_size, position, batch] = 1.0

    return in_tensor, out_tensor


def read_buffer(buffer, pointer):
    """
    Read an element from buffer.

    :param buffer: [*, length, batch_size]
    :param pointer: [length, batch_size]
    :return: element at pointer [*, batch_size]
    """
    with tf.name_scope("read_buffer"):
        weighted = buffer * pointer
        reduced = tf.reduce_sum(weighted, 1)
    return reduced


def write_buffer(buffer, pointer, value):
    """
    Write an element into a buffer.

    :param buffer: [width, length, batch_size]
    :param pointer: [length, batch_size]
    :param value: [width, batch_size]
    :return: new buffer with element written into the pointer location.
    """
    with tf.name_scope("write_buffer"):
        matrix_of_current_value = buffer * pointer  # [width, length, batch_size]
        expanded_new_value = tf.expand_dims(value, 1)  # [width, 1, batch_size]
        matrix_of_new_value = expanded_new_value * pointer
        result = buffer - matrix_of_current_value + matrix_of_new_value
    return result


def binarize(original, threshold=0.0, temperature=1.0, left=0.5, right=0.5):
    """
    Map real value into [0, 1] range using either sigmoids or piecewise linear functions.

    :param original: original tensor
    :param threshold: threshold at which the (0 temperature) score should be 0.5
    :param temperature: temperature of the sigmoid
    :param left: at 0 temperature, where should the ramp begin relative to threshold
    :param right: at 0 temperature, where should the ramp end relative to threshold
    :return:mapping of original tensor into [0,1] range
    """

    with tf.name_scope("binarize"):
        if temperature > 0.0:
            return tf.sigmoid((original - threshold) / temperature, name="sigmoid")
        else:
            with tf.name_scope("piecewise"):
                begin = threshold - left
                end = threshold + right
                scale = 1.0 / (end - begin)
                res = tf.minimum(tf.maximum(0.0, (original - begin) * scale), 1.0)
                return res


def transform_width(matrix, width_in, width_out):
    """
    Need to transform the width of a matrix when D and R have different widths.
    WARNING: this clips values when going from a longer to a shorter size
             and can result in bad (clipped) values

    :param width_out: input width
    :param width_in: output width
    :param matrix: [width_in, batch_size]
    :return: matrix [batch_size, width_out]
    """
    if width_in > width_out:
        return matrix[0:width_out, :]
    elif width_in < width_out:
        return tf.pad(matrix, [[0, width_out - width_in], [0, 0]])
    else:
        return matrix


def pretty_print_buffer(buffer):
    """
    Transpose buffer into [batch_size, length, width] for pretty print

    :param buffer: input buffer in [width, length, batch_size]
    :return: buffer in [batch_size, length, width] shape.
    """
    return tf.transpose(buffer, [2, 1, 0])


def pretty_print_value(value):
    """
    Transpose the pointer value around for pretty printout

    :param value: input pointer in [width, batch_size]
    :return: pointer in [batch_size, width]
    """
    return tf.transpose(value)


def print_dsm_state_np(data_stack, data_stack_pointer,
                       return_stack=None, return_stack_pointer=None,
                       heap=None, pc=None, file=sys.stdout, interpreter=None):
    """
    Just pretty printing the whole state out
    """
    def print_out(text):
        print(text, file=file)

    print_out("-" * 50)
    print_out("D:")
    print_out(np.transpose(data_stack, [2, 1, 0]))
    print_out("DP:")
    print_out(np.transpose(data_stack_pointer))

    if return_stack is not None:
        print_out("-" * 50)
        print_out("R:")
        print_out(np.transpose(return_stack, [2, 1, 0]))
        print_out("RP:")
        print_out(np.transpose(return_stack_pointer))

    if pc is not None:
        print_out("-" * 50)
        print_out("PC:")
        print_out(np.transpose(pc))
        # TODO: didn't check for a possible lack of the interpreter
        for batch in range(0, interpreter.batch_size):
            print("Batch {}".format(batch))
            for word_index in range(0, interpreter.code_size):
                score = pc[word_index, batch]
                if score > 0.5:
                    print("{} {}".format(score, interpreter.words[
                        interpreter.final_code[batch][word_index]
                    ]))

    if heap is not None:
        print_out("-" * 50)
        print_out("HEAP:")
        print_out(np.transpose(heap, [2, 1, 0]))


class CachedFunction:
    """ Simply cache functions """
    def __init__(self, f):
        self.f = f
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            # print("Cached!")
            return self.cache[args]
        else:
            result = self.f(*args)
            self.cache[args] = result
            return result


# TODO: unused for now
def cache_function(f):
    """
    Caches a function
    :param f: function to cache
    :return: cached function.
    """
    cache = {}

    def cached(*args):
        if args in cache:
            # print("Cached!")
            return cache[args]
        else:
            result = f(*args)
            cache[args] = result
            return result

    return cached


class DSMState:
    def __init__(self, code_size,
                 data_stack, data_stack_pointer,
                 return_stack, return_stack_pointer,
                 heap, pc, code,
                 vocab: AssembledVocab,
                 labels, constants,
                 return_width=None,
                 temperature=0.0,
                 do_normalise_pointer=True):
        """
        Create an extensible DSM.
        :param data_stack: tensor representation data stack buffer
                           [value_size, stack_size, batch_size]
        :param data_stack_pointer: top pointer [stack_size, batch_size]
        :param return_stack: tensor representation return stack buffer
                             [code_size, return_stack_size, batch_size]
        :param return_stack_pointer: top pointer for return stack [return_stack_size, batch_size]
        :param heap: heap tensor [value_size, heap_size, batch_size]
        :param pc: [code_size, batch_size]
        :param code: [vocab_size, code_size, batch_size]
        :param vocab: vocabulary of assembled forth words
        :param labels: [code_size, num_labels, batch_size] for each batch a list of
                                                           code pointers to jump to.
        :param constants: [value_size, num_constants, batch_size]
        :param temperature: temperature used when mapping values into [0,1]
        :param do_normalise_pointer: should pointers be normalised after each step.
        """
        self.code_size = code_size
        self.do_normalise_pointer = do_normalise_pointer
        self.batch_size = data_stack.get_shape()[2].value
        self.value_size = data_stack.get_shape()[0].value
        self.stack_size = data_stack.get_shape()[1].value
        self.return_stack_size = return_stack.get_shape()[1].value
        self.return_width = return_width if return_width else return_stack.get_shape()[0].value

        self.temperature = temperature
        self.constants = constants
        self.labels = labels
        self.code = code
        self.pc = pc
        self.data_stack = data_stack
        self.data_stack_pointer = data_stack_pointer
        self.return_stack = return_stack
        self.return_stack_pointer = return_stack_pointer
        self.heap = heap
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.inc_matrices = CachedFunction(lambda size, i: create_inc_matrix(size, i))

        self.one_hot_add_matrix = create_alg_op_matrix(self.value_size, 'add')
        self.one_hot_sub_matrix = create_alg_op_matrix(self.value_size, 'sub')
        self.one_hot_mul_matrix = create_alg_op_matrix(self.value_size, 'mul')
        self.one_hot_div_matrix = create_alg_op_matrix(self.value_size, 'div')


        self.pc_inc_matrix_pairs = CachedFunction(
            lambda positions: create_pc_inc_matrix_pair(positions, code_size, self.batch_size))
        # self.evaluate = cache_function(self.evaluate_force)
        self.evaluate = CachedFunction(self.evaluate_force)


        def one_hot(active, dim, value=1.0):
            """ Returns a one-hot vector, with value on active index """
            # x = np.zeros(dim, dtype='float32')
            # x[active] = 1.0
            return [value if i == active else 0.0 for i in range(0, dim)]

        self.zero = tf.expand_dims(
            tf.constant(one_hot(0, self.value_size), dtype=tf.float32), 1, name="zero")
        self.one = tf.expand_dims(
            tf.constant(one_hot(1, self.value_size), dtype=tf.float32), 1, name="one")

        self.integer_weights = tf.expand_dims(tf.constant(
            [i for i in range(0, self.value_size)], dtype=tf.float32), 1, name="integer_weights")
        # [value_size, 1]

    def execute(self, word_weights):
        """
        Execute word (distribution).
        :param word_weights: [batch_size, vocab_size]
        :return: new ExtensibleDSM state after transition.
        """
        with tf.name_scope("execute"):
            next_states = [word(self) for word in self.vocab.words]
            result = merge_dsms(self, next_states, word_weights, self.do_normalise_pointer)
            return result

    def step(self, name=None):
        """
        Execute one step of the DSM using the code at PC.
        :return: next state.
        """
        # logging.debug("Creating DSM after step")
        if name is None:
            name = "train_step"
        with tf.name_scope(name):
            word_weights = self.evaluate(sym.selected_word())
            return self.execute(word_weights)

    def copy(self):
        """
        Copies this DSM, and clears cache.
        :return: shallow copy of DSM, with cleared cache.
        """
        result = copy.copy(self)
        # logging.debug("Copying DSM with {} cached computations".format(len(self.evaluate.cache)))
        result.evaluate = CachedFunction(result.evaluate_force)
        return result

    def evaluate_force(self, term):
        """
        Evaluates a symbolic term against the state of the machine.

        :param term: the symbolic term to evaluate
        :return: a tensorflow node that represents the computation for the term.
        """
        if term == sym.data_stack_pointer():
            return self.data_stack_pointer

        if term == sym.return_stack_pointer():
            return self.return_stack_pointer

        elif term == sym.heap():
            return self.heap

        elif term == sym.data_stack():
            return self.data_stack

        elif term == sym.return_stack():
            return self.return_stack

        elif term == sym.pc():
            return self.pc

        elif symbol_match(term, sym.next_pc()):
            with tf.name_scope("NEXT_PC"):
                positions_in_batch = term[1]
                inc_in, inc_out = self.pc_inc_matrix_pairs(positions_in_batch)
                current_pc = self.evaluate(sym.pc())  # [code_size, batch_size ]

                matched = tf.reduce_sum(current_pc * inc_in, 1)  # [code_size, batch_size]
                mapped = matched * inc_out  # [code_size, code_size, batch_size]
                merged = tf.reduce_sum(mapped, 1)
                return merged
            # return self.evaluate(
            #     sym.inc(sym.pc(), 1, self.code_size))
            # TODO should use the OC transition matrix for the batch

        elif term == sym.one():
            return self.one

        elif term == sym.zero():
            return self.zero

        elif term == sym.selected_word():
            with tf.name_scope("SEL_WORD"):
                return tf.reduce_sum(self.pc * self.code, 1)

        elif symbol_match(term, sym.write_buffer()):
            buffer, addr, value = [self.evaluate(t) for t in term[1:]]
            return write_buffer(buffer, addr, value)

        elif symbol_match(term, sym.eq()):
            with tf.name_scope("EQUALS"):
                arg1, arg2 = term[1:]
                node1 = self.evaluate(arg1)
                node2 = self.evaluate(arg2)
                score = tf.reduce_sum(node1 * node2, 0, keep_dims=True)
                prob = binarize(score, 0.5, self.temperature)
                return prob * self.one + (1.0 - prob) * self.zero

        elif symbol_match(term, sym.gt()):
            with tf.name_scope("GREATER_THAN"):
                arg1, arg2 = term[1:]
                node1 = self.evaluate(arg1)  # [value_size, batch_size]
                node2 = self.evaluate(arg2)  # [value_size, batch_size]
                weighted1 = self.integer_weights * node1
                weighted2 = self.integer_weights * node2
                expected1 = tf.reduce_sum(weighted1, 0, keep_dims=True)
                expected2 = tf.reduce_sum(weighted2, 0, keep_dims=True)
                prob = binarize(expected1 - expected2, 0.5, self.temperature)
                return prob * self.one + (1.0 - prob) * self.zero

        elif symbol_match(term, sym.vector_plus()):
            with tf.name_scope("v_plus"):
                arg1, arg2 = term[1:]
                node1 = self.evaluate(arg1)
                node2 = self.evaluate(arg2)
            return node1 + node2

        elif symbol_match(term, sym.vector_minus()):
            with tf.name_scope("v_minus"):
                arg1, arg2 = term[1:]
                node1 = self.evaluate(arg1)
                node2 = self.evaluate(arg2)
            return node1 - node2

        elif symbol_match(term, sym.vector_times()):
            with tf.name_scope("v_times"):
                arg1, arg2 = term[1:]
                node1 = self.evaluate(arg1)
                node2 = self.evaluate(arg2)
                return node1 * node2

        # TODO FIX DIVISION!
        elif symbol_match(term, sym.vector_divide()):
            with tf.name_scope("v_div"):
                arg1, arg2 = term[1:]
                node1 = self.evaluate(arg1)
                node2 = self.evaluate(arg2)
                return node1 / (node2 + eps)

        elif symbol_match(term, sym.one_hot_add()):
            with tf.name_scope("one_hot_add"):
                arg1, arg2 = term[1:]
                node1 = self.evaluate(arg1) # [value_size, batch_size]
                node2 = self.evaluate(arg2) # [value_size, batch_size]
                add_reshape = tf.reshape(self.one_hot_add_matrix,
                                         [self.value_size, self.value_size * self.value_size])
                temp = tf.matmul(tf.transpose(node1), add_reshape)
                temp_reshape = tf.reshape(temp, [self.batch_size, self.value_size, self.value_size])
                result = tf.transpose(tf.reshape(tf.batch_matmul(tf.expand_dims(tf.transpose(node2), 1), temp_reshape), shape=[self.batch_size, self.value_size]))
                return result

        elif symbol_match(term, sym.one_hot_sub()):
            with tf.name_scope("one_hot_sub"):
                arg1, arg2 = term[1:]
                node1 = self.evaluate(arg1) # [value_size, batch_size]
                node2 = self.evaluate(arg2) # [value_size, batch_size]
                add_reshape = tf.reshape(self.one_hot_sub_matrix, [self.value_size, self.value_size * self.value_size])
                temp = tf.matmul(tf.transpose(node1), add_reshape)
                temp_reshape = tf.reshape(temp, [self.batch_size, self.value_size, self.value_size])
                result = tf.transpose(tf.reshape(tf.batch_matmul(tf.expand_dims(tf.transpose(node2), 1), temp_reshape), shape=[self.batch_size, self.value_size]))
                return result

        elif symbol_match(term, sym.one_hot_mul()):
            with tf.name_scope("one_hot_mul"):
                arg1, arg2 = term[1:]
                node1 = self.evaluate(arg1) # [value_size, batch_size]
                node2 = self.evaluate(arg2) # [value_size, batch_size]
                add_reshape = tf.reshape(self.one_hot_mul_matrix, [self.value_size, self.value_size * self.value_size])
                temp = tf.matmul(tf.transpose(node1), add_reshape)
                temp_reshape = tf.reshape(temp, [self.batch_size, self.value_size, self.value_size])
                result = tf.transpose(tf.reshape(tf.batch_matmul(tf.expand_dims(tf.transpose(node2), 1), temp_reshape), shape=[self.batch_size, self.value_size]))
                return result

        elif symbol_match(term, sym.one_hot_div()):
            with tf.name_scope("one_hot_div"):
                arg1, arg2 = term[1:]
                node1 = self.evaluate(arg1) # [value_size, batch_size]
                node2 = self.evaluate(arg2) # [value_size, batch_size]
                add_reshape = tf.reshape(self.one_hot_div_matrix, [self.value_size, self.value_size * self.value_size])
                temp = tf.matmul(tf.transpose(node1), add_reshape)
                temp_reshape = tf.reshape(temp, [self.batch_size, self.value_size, self.value_size])
                result = tf.transpose(tf.reshape(tf.batch_matmul(tf.expand_dims(tf.transpose(node2), 1), temp_reshape), shape=[self.batch_size, self.value_size]))
                return result


        elif symbol_match(term, sym.choice()):
            with tf.name_scope("branch"):
                # TODO: should use branch point specific PC transition matrix
                test, true_branch, false_branch = term[1:]
                with tf.name_scope("IF_condition"):
                    node_test = self.evaluate(test)  # [value_size, batch_size]
                with tf.name_scope("THEN_branch"):
                    node_true_branch = self.evaluate(true_branch)  # [code_size, batch_size]
                with tf.name_scope("ELSE_branch"):
                    node_false_branch = self.evaluate(false_branch)  # [code_size, batch_size]
                neg_prob = node_test[0, :]  # [batch_size]?
                with tf.name_scope("IF_THEN_ELSE"):
                    result = neg_prob * node_false_branch + (1.0 - neg_prob) * node_true_branch
                    return result

        elif symbol_match(term, sym.label()):
            label = term[1]
            with tf.name_scope("LABEL_{}".format(label)):
                return self.labels[:, label, :]

        elif symbol_match(term, sym.store()):
            with tf.name_scope("STORE"):
                value, addr, buffer = term[1:]  # TODO: if addr is a constant, stitch heap
                buffer_node = self.evaluate(buffer)
                value_node = self.evaluate(value)
                return write_buffer(buffer_node, self.evaluate(addr), value_node)

        elif symbol_match(term, sym.fetch()):
            with tf.name_scope("FETCH"):
                addr, buffer = term[1:]  # TODO: if addr is a constant, simply return slice of heap
                return read_buffer(self.evaluate(buffer), self.evaluate(addr))

        elif symbol_match(term, sym.constant()):
            constant = term[1]
            with tf.name_scope("CONSTANT_{}_".format(constant)):
                return self.constants[:, constant, :]

        elif symbol_match(term, sym.convert_data_to_return()):
            with tf.name_scope("D_to_R"):
                input_term = term[1]
                input_evaluated = self.evaluate(input_term)
                return transform_width(input_evaluated, self.value_size, self.return_width)

        elif symbol_match(term, sym.convert_return_to_data()):
            with tf.name_scope("R_to_D"):
                input_term = term[1]
                input_evaluated = self.evaluate(input_term)
                return transform_width(input_evaluated, self.return_width, self.value_size)

        elif symbol_match(term, sym.convert_pc_to_return()):
            with tf.name_scope("PC_to_R"):
                input_term = term[1]
                input_evaluated = self.evaluate(input_term)
                result = transform_width(input_evaluated, self.code_size, self.return_width)
                return result

        elif symbol_match(term, sym.convert_return_to_pc()):
            with tf.name_scope("R_to_PC"):
                input_term = term[1]
                input_evaluated = self.evaluate(input_term)
                return transform_width(input_evaluated, self.return_width, self.code_size)

        elif symbol_match(term, sym.inc_return_value()):
            with tf.name_scope("INC_RETURN_VALUE"):
                return self.evaluate(sym.inc(term[1], term[2], self.return_width))

        elif symbol_match(term, sym.inc()):
            value, amount, dim = term[1:]
            name = "INC"
            if amount < 0:
                name = "DEC"
            with tf.name_scope("{0}".format(name)):
                if symbol_match(value, sym.inc()):
                    assert value[3] == dim
                    return self.evaluate(sym.inc(value[1], amount + value[2], value[3]))
                else:
                    value_node = self.evaluate(value)  # [dim, batch_size]
                    if amount == 0:
                        return value_node

                    # get transition matrix
                    transition = self.inc_matrices(dim, amount)  # [dim, dim]
                    return tf.matmul(transition, value_node)

        elif symbol_match(term, sym.data_stack_elem()):
            index = term[1]
            with tf.name_scope("D{}".format(index)):
                pointer = self.evaluate(sym.inc(sym.data_stack_pointer(), index, self.stack_size))
                elem = read_buffer(self.data_stack, pointer)
                return elem

        elif symbol_match(term, sym.return_stack_elem()):
            index = term[1]
            with tf.name_scope("R{}".format(index)):
                pointer = self.evaluate(sym.inc(sym.return_stack_pointer(), index, self.stack_size))
                elem = read_buffer(self.return_stack, pointer)
                return elem

        else:
            raise KeyError(term)


eps = 0.0000000000000001


class SymbolicDSMWord(AssembledWord):
    """
    A word that transforms the DSM state based on the state of a symbolic DSM.
    """

    def __init__(self, symbolic_dsm: sym.SymbolicDSM, word):
        super().__init__()
        self.name = word
        self.symbolic_dsm = symbolic_dsm
        self.executed = False
        self.word = word

    def __str__(self):
        return "{}: {}".format(self.name, self.symbolic_dsm)

    def __call__(self, dsm: DSMState):
        if not self.executed:
            self.symbolic_dsm.set_positions_in_batches(self.indices_in_batches)
            self.symbolic_dsm.execute_im(self.word)
            self.executed = True
        result = dsm.copy()
        # set the new stack pointer
        result.data_stack_pointer = dsm.evaluate(
            sym.inc(sym.data_stack_pointer(), self.symbolic_dsm.data_stack_top, dsm.stack_size))

        # update elements in the data stack buffer
        current_sym_data_stack = sym.data_stack()
        for i in range(self.symbolic_dsm.min_data_depth, self.symbolic_dsm.data_stack_top + 1):
            sym_value = self.symbolic_dsm.data_stack[i]
            if sym_value != sym.data_stack_elem(i):
                sym_addr = sym.inc(sym.data_stack_pointer(), i, dsm.stack_size)
                current_sym_data_stack = sym.write_buffer(current_sym_data_stack,
                                                          sym_addr, sym_value)
                result.data_stack = dsm.evaluate(current_sym_data_stack)

                # node_value = dsm.evaluate(sym_value)
                # node_pointer = dsm.evaluate(sym.inc(sym.data_stack_pointer(), i, dsm.stack_size))
                # result.data_stack = write_buffer(result.data_stack, node_pointer, node_value)

        # set the new return stack pointer
        result.return_stack_pointer = dsm.evaluate(
            sym.inc(sym.return_stack_pointer(), self.symbolic_dsm.return_stack_top, dsm.stack_size))

        current_sym_return_stack = sym.return_stack()
        # update elements in the return stack buffer
        for i in range(self.symbolic_dsm.min_return_depth, self.symbolic_dsm.return_stack_top + 1):
            sym_value = self.symbolic_dsm.return_stack[i]
            if sym_value != sym.return_stack_elem(i):
                sym_addr = sym.inc(sym.return_stack_pointer(), i, dsm.stack_size)
                current_sym_return_stack = sym.write_buffer(current_sym_return_stack,
                                                            sym_addr, sym_value)
                result.return_stack = dsm.evaluate(current_sym_return_stack)
                # node_value = dsm.evaluate(sym_value)
                # node_pointer = dsm.evaluate(sym.inc(sym.return_stack_pointer(), i, dsm.stack_size))
                # result.return_stack = write_buffer(result.return_stack, node_pointer, node_value)

        # update program counter
        result.pc = dsm.evaluate(self.symbolic_dsm.pc)

        # update heap
        result.heap = dsm.evaluate(self.symbolic_dsm.heap)

        return result


class DSMEncoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, dsm: DSMState):
        """
        Encode the input DSM.
        :param dsm: the input DSM state.
        :return: a [output_dim, batch_size] representation of `dsm`.
        """
        pass


class DSMDecoder(metaclass=abc.ABCMeta):
    def __init__(self):
        self.word = None

    def set_word(self, word: AssembledWord):
        self.word = word

    @abc.abstractmethod
    def __call__(self, dsm, hidden_repr):
        pass

    @abc.abstractmethod
    def input_dim(self):
        pass


class StaticEncoder(DSMEncoder):
    """
    Encodes every DSM state using the same static vector.
    """

    def __init__(self, output_dim, init_weight_stddev):
        self.output_dim = output_dim
        # self.bias = tf.Variable(tf.zeros([output_dim, 1]))
        # self.bias = tf.Variable([[10.],[-10.]])
        # TODO experiment with variance
        self.bias = tf.Variable(tf.random_normal([output_dim, 1],
                                                 stddev=init_weight_stddev), name="bias")
        # self.bias = tf.Variable(tf.ones([output_dim, 1]), name="bias")

    def __call__(self, dsm: DSMState):
        with tf.name_scope("static_decoder"):
            # TODO: this could be done only once if we know batch_size
            tiled = tf.tile(self.bias, [1, dsm.batch_size])
        return tiled


class Transformer(DSMEncoder):
    def __init__(self, encoder: DSMEncoder, transformation):
        self.encoder = encoder
        self.transformation = transformation

    def __call__(self, dsm: DSMState):
        return self.transformation(self.encoder(dsm))


class ObserveEncoder(DSMEncoder):
    """
    Observes a sequence of DSM elements such as "value at top of data stack" indicated as ('D',0).
    """

    def __init__(self, dsm_elements, output_dim, value_size, return_width, init_weight_stddev):
        assert len(dsm_elements) == len(set(dsm_elements))
        self.output_dim = output_dim
        self.dsm_elements = dsm_elements
        # TODO experiment with variance
        self.bias = tf.Variable(tf.random_normal(
            [output_dim, 1], stddev=init_weight_stddev), name="bias")
        self.input_dim = 0
        for elem in self.dsm_elements:
            if elem[0] == 'D':
                self.input_dim += value_size
            elif elem[0] == 'R':
                self.input_dim += return_width
            elif elem[0] == 'H':
                self.input_dim += return_width

        # TODO experiment with variance
        self.weights = tf.Variable(tf.random_normal(
            [self.output_dim, self.input_dim], stddev=init_weight_stddev), name="weights")

    def __call__(self, dsm: DSMState):
        # we concat all stack elements
        with tf.name_scope("OBSERVE"):
            to_concat = []
            for elem in self.dsm_elements:
                if elem[0] == 'D':
                    stack_index = elem[1]
                    # [value_size, batch_size]
                    to_concat.append(dsm.evaluate(sym.data_stack_elem(stack_index)))
                elif elem[0] == 'R':
                    stack_index = elem[1]
                    # [value_size, batch_size]
                    to_concat.append(dsm.evaluate(sym.return_stack_elem(stack_index)))
                elif elem[0] == 'H':
                    heap_index = elem[1]
                    heap_element = dsm.heap[:, heap_index, :]
                    to_concat.append(heap_element)  # [value_size, batch_size]
            input_repr = tf.concat(0, to_concat, name="concat")  # [input_dim, batch_size]
            with tf.name_scope("linear"):
                result = tf.matmul(self.weights, input_repr) + self.bias
            return result


class ConjoinEncoder(DSMEncoder):
    """
    Observes a sequence of DSM elements such as "value at top of data stack" indicated as ('D',0).
    """

    def __init__(self, dsm_elements, output_dim, value_size, return_width, init_weight_stddev):
        assert len(dsm_elements) == len(set(dsm_elements))
        self.output_dim = output_dim
        self.dsm_elements = dsm_elements
        self.bias = tf.Variable(tf.random_normal([output_dim, 1], stddev=init_weight_stddev), name="conjoin_bias")
        self.input_dim = 0
        for elem in self.dsm_elements:
            if elem[0] == 'D':
                self.input_dim += value_size
            elif elem[0] == 'R':
                self.input_dim += return_width
            elif elem[0] == 'H':
                self.input_dim += return_width

        self.input_dim = self.input_dim * self.input_dim
        self.weights = tf.Variable(tf.random_normal([self.output_dim, self.input_dim], stddev=init_weight_stddev), name="conjoin_weight")

    def __call__(self, dsm: DSMState):
        # we concat all stack elements
        to_concat = []
        for elem in self.dsm_elements:
            if elem[0] == 'D':
                stack_index = elem[1]
                to_concat.append(dsm.evaluate(sym.data_stack_elem(stack_index)))  # [value_size, batch_size]
            elif elem[0] == 'R':
                stack_index = elem[1]
                to_concat.append(dsm.evaluate(sym.return_stack_elem(stack_index)))  # [value_size, batch_size]
            elif elem[0] == 'H':
                heap_index = elem[1]
                heap_element = dsm.heap[:, heap_index, :]
                to_concat.append(heap_element)  # [value_size, batch_size]
        input_repr = tf.concat(0, to_concat)  # [input_dim, batch_size]

        arg1 = tf.expand_dims(input_repr, 0)
        arg2 = tf.expand_dims(input_repr, 1)
        conjoined = arg1 * arg2
        flattened = tf.reshape(conjoined, [self.input_dim, dsm.batch_size])
        result = tf.matmul(self.weights, flattened) + self.bias
        return result


class ChoiceDecoder(DSMDecoder):
    """
    Produces the next state as a weighted combination of a set of word choices.
    """

    def __init__(self, choice_words):
        super().__init__()
        self.choice_words = choice_words

    def set_word(self, word):
        super().set_word(word)
        for choice in self.choice_words:
            choice.indices_in_batches = word.indices_in_batches

    def __call__(self, dsm: DSMState, hidden_repr):
        # hidden : [num_choices, batch_size]
        with tf.name_scope("choice_decoder"):
            # TODO peak this guy out with entropy loss?
            with tf.name_scope("choice_attention"):
                # exp(logits[i, j]) / sum_j(exp(logits[i, j]))
                attention = tf.transpose(tf.nn.softmax(tf.transpose(hidden_repr)))
            choice_states = [word(dsm) for word in self.choice_words]
            merged = merge_dsms(dsm, choice_states, attention, do_normalise=True)
        return merged

    def input_dim(self):
        return len(self.choice_words)


class SampleDecoder(DSMDecoder):
    """
    Produces the next state by sampling a word to execute.
    """

    def __init__(self, choice_words):
        super().__init__()
        self.choice_words = choice_words
        self.last_sample_node = None
        self.sample_values = []
        self.sample_placeholders = []
        self.in_training = False
        self.distributions = []

    def clear_sample_history(self):
        self.sample_values = []

    def set_word(self, word):
        super().set_word(word)
        for choice in self.choice_words:
            choice.indices_in_batches = word.indices_in_batches

    def set_mode(self, training=False):
        self.in_training = training

    def remember_sample_value(self, value):
        self.sample_values.append(value)

    def current_feed_dict(self, length=None):
        num_samples = min(length, len(self.sample_placeholders)) if length is not None else len(
            self.sample_placeholders)
        assert len(self.sample_values) >= num_samples
        result = {}
        for i in range(0, num_samples):
            result[self.sample_placeholders[i]] = self.sample_values[i]
        return result

    def get_all_losses(self, downstream_loss_sample, length=None):
        num_steps = min(len(self.sample_placeholders), length) if length is not None else len(self.sample_placeholders)
        losses = []
        for i in range(0, num_steps):
            sample = self.sample_placeholders[i]  # [num_choices+1, batch_size]
            dist = self.distributions[i]  # [num_choices+1, batch_size]
            sum = tf.reduce_sum(sample * dist, 0)  # [batch_size]
            log_prob = tf.log(sum)  # [batch_size]
            scaled_by_downstream_loss = log_prob * downstream_loss_sample  # [batch_size]
            losses.append(scaled_by_downstream_loss)
        packed = tf.pack(losses)  # [num_steps, batch]
        return packed

    def __call__(self, dsm: DSMState, hidden_repr):
        self.pc_weighting = np.zeros((dsm.code_size, dsm.batch_size))
        for batch, indices in self.word.indices_in_batches.items():
            for index in indices:
                self.pc_weighting[index, batch] = 1.0

        mass_on_decoder = tf.reduce_max(dsm.pc * self.pc_weighting, 0, keep_dims=True)  # [1, batch_size]

        # hidden : [num_choices, batch_size]
        attention = tf.transpose(tf.nn.softmax(tf.transpose(hidden_repr)))

        # split remaining mass onto choices
        reduced_attention = mass_on_decoder * attention

        # distribution over NOP and choices
        dist = tf.concat(0, [1.0 - mass_on_decoder, reduced_attention])

        if self.in_training:
            sampled = tf.placeholder(tf.float32, [len(self.choice_words) + 1, dsm.batch_size])
            self.sample_placeholders.append(sampled)
            self.distributions.append(dist)
        else:

            # TODO: could also use sym.selected_word() if we know the index of the owning word

            # sample from it # TODO avoid transpose by improving batch_sample_with_temperature
            import nam.util.sampling as sampling
            sampled = tf.transpose(sampling.batch_sample_with_temperature(tf.transpose(dist)))

            self.last_sample_node = sampled
            # sampled = tf.transpose(sampling.batch_sample_with_temperature(tf.transpose(attention)))

        # choice_states = [word(dsm) for word in self.choice_words]
        choice_states = [dsm] + [word(dsm) for word in self.choice_words]
        merged = merge_dsms(dsm, choice_states, sampled)

        return merged

    def input_dim(self):
        return len(self.choice_words)


class PermuteDecoder(DSMDecoder):
    """
    Produces the next state by permuting machine elements such as the stack content.
    """

    def __init__(self, dsm_elements):
        super().__init__()
        import math
        self.dsm_elements = dsm_elements
        self._input_dim = math.factorial(len(dsm_elements))

    def __call__(self, dsm: DSMState, hidden_repr):
        # hidden : [input_dim, batch_size]

        def create_all_permutations(lhs_elems, rhs_elems):
            # for each lhs element, assign it to each rhs elem, and then assign remaining elements
            if len(lhs_elems) == 0:
                return [[]]
            else:
                lhs = lhs_elems[0]
                all = []
                for rhs in rhs_elems:
                    previous = create_all_permutations(lhs_elems[1:], [r for r in rhs_elems if r != rhs])
                    result = [[(lhs, rhs)] + prev for prev in previous]
                    all.extend(result)
                return all

        with tf.name_scope("permute_decoder"):
            permutations = create_all_permutations(self.dsm_elements, self.dsm_elements)
            assert len(permutations) == self._input_dim
            choices = []
            for permutation in permutations:
                # permutation is a list of assignment tuples
                result = dsm.copy()
                for (lhs, rhs) in permutation:
                    id_lhs, stack_index_lhs = lhs
                    id_rhs, stack_index_rhs = rhs
                    value = None
                    width_rhs = None
                    if id_rhs == 'D':
                        value = dsm.evaluate(sym.data_stack_elem(stack_index_rhs))
                        width_rhs = dsm.value_size
                    elif id_rhs == 'R':
                        value = dsm.evaluate(sym.return_stack_elem(stack_index_rhs))
                        width_rhs = dsm.return_width
                    if id_lhs == 'D':
                        pointer = dsm.evaluate(sym.inc(sym.data_stack_pointer(),
                                                       stack_index_lhs, dsm.stack_size))
                        width_lhs = dsm.value_size
                        result.data_stack = write_buffer(result.data_stack, pointer,
                                                         transform_width(value, width_rhs,
                                                                         width_lhs))
                    elif id_lhs == 'R':
                        pointer = dsm.evaluate(sym.inc(sym.return_stack_pointer(),
                                                       stack_index_lhs, dsm.stack_size))
                        width_lhs = dsm.return_width
                        result.return_stack = write_buffer(result.return_stack, pointer,
                                                           transform_width(value, width_rhs,
                                                                           width_lhs))
                choices.append(result)

            with tf.name_scope("permute_attention"):
                attention = tf.transpose(tf.nn.softmax(tf.transpose(hidden_repr)))
            merged = merge_dsms(dsm, choices, attention, do_normalise=True)

            with_tuples = [(key, tuple(value)) for key, value in self.word.indices_in_batches.items()]
            indices_in_batches = frozenset(with_tuples)
            merged.pc = dsm.evaluate(sym.next_pc(indices_in_batches))
            return merged

    def input_dim(self):
        return self._input_dim


class ManipulateDecoder(DSMDecoder):
    """
    Produces the next state by directly manipulating machine elements such as the stack content.
    """

    def __init__(self, dsm_elements, value_size, return_width):
        super().__init__()
        self.return_width = return_width
        assert len(dsm_elements) == len(set(dsm_elements))
        self.value_size = value_size
        self.dsm_elements = dsm_elements
        self._input_dim = 0
        self.offsets = []
        for elem in self.dsm_elements:
            self.offsets.append(self._input_dim)
            if elem[0] == 'D':
                self._input_dim += value_size
            elif elem[0] == 'H':
                self._input_dim += value_size
            elif elem[0] == 'R':
                self._input_dim += return_width

    def __call__(self, dsm: DSMState, hidden_repr):
        # hidden : [input_dim, batch_size]
        with tf.name_scope("MANIPULATE"):
            result = dsm.copy()
            for elem_id, elem in enumerate(self.dsm_elements):
                if elem[0] == 'R':
                    id, stack_index = elem
                    with tf.name_scope("R{}".format(stack_index)):
                        node_value = hidden_repr[self.offsets[elem_id]:self.offsets[elem_id] + self.return_width, :]
                        node_value = tf.transpose(tf.nn.softmax(tf.transpose(node_value)))
                        node_pointer = dsm.evaluate(sym.inc(sym.return_stack_pointer(), stack_index, dsm.stack_size))
                        result.return_stack = write_buffer(result.return_stack, node_pointer, node_value)
                elif elem[0] == 'D':
                    id, stack_index = elem
                    with tf.name_scope("D{}".format(stack_index)):
                        node_value = hidden_repr[self.offsets[elem_id]:self.offsets[elem_id] + self.value_size, :]
                        node_value = tf.transpose(tf.nn.softmax(tf.transpose(node_value)))
                        node_pointer = dsm.evaluate(sym.inc(sym.data_stack_pointer(), stack_index, dsm.stack_size))
                        result.data_stack = write_buffer(result.data_stack, node_pointer, node_value)
                elif elem[0] == 'H':
                    _, heap_index = elem
                    with tf.name_scope("H{}".format(stack_index)):
                        node_value = hidden_repr[self.offsets[elem_id]:self.offsets[elem_id] + self.value_size, :]
                        before = result.dsm.heap[:, 0:heap_index, :]
                        after = result.dsm.heap[:, heap_index + 1:, :]
                        insert = tf.expand_dims(node_value, 1)  # [value_size, 1, batch_size]
                        result.heap = tf.concat(1, [before, insert, after])

            with_tuples = [(key, tuple(value)) for key, value in self.word.indices_in_batches.items()]
            indices_in_batches = frozenset(with_tuples)

            result.pc = dsm.evaluate(sym.next_pc(indices_in_batches))
            return result

    def input_dim(self):
        return self._input_dim

    def __str__(self):
        return "MANIPULATE {}".format(self.dsm_elements)


class EncoderDecoderWord(AssembledWord):
    """
    Constructs a [enc.input_dim(), batch_size] representation of the input DSM state
    using the encoder, and then produces a new state using the decoder.
    """

    def __init__(self, enc: DSMEncoder, dec: DSMDecoder):
        super().__init__()
        self.dec = dec
        self.enc = enc
        self.dec.set_word(self)

    def __call__(self, dsm: DSMState):
        with tf.name_scope("encoder_decoder"):
            with tf.name_scope("encoder"):
                hidden_representation = self.enc(dsm)
            with tf.name_scope("decoder"):
                next_state = self.dec(dsm, hidden_representation)
        return next_state

    def __str__(self):
        return "{} > {} -> {}".format(str(self.indices_in_batches), self.enc, self.dec)


def main():
    value_size = 4
    code_size = 10
    stack_size = 6
    batch_size = 2
    heap_size = 5
    max_labels = 10
    max_constants = 10
    sess = tf.Session()
    vocab = AssembledVocab()
    # vocab.add_word(Halt())
    sym_dsm = sym.SymbolicDSM(stack_size, stack_size, heap_size, value_size)
    vocab.add_word(SymbolicDSMWord(sym_dsm.copy().halt(), ('DUP',)))
    dsm = DSMState(
        data_stack=create_diag_buffer(stack_size, value_size, batch_size),
        data_stack_pointer=create_pointer(stack_size, batch_size, -1),
        return_stack=create_buffer(stack_size, code_size, batch_size),
        return_stack_pointer=create_pointer(stack_size, batch_size, -1),
        heap=create_buffer(heap_size, value_size, batch_size, 1.0),
        pc=create_pointer(code_size, batch_size, 0),
        code=create_diag_buffer(code_size, len(vocab), batch_size),
        vocab=vocab,
        labels=create_diag_buffer(max_labels, code_size, batch_size),
        constants=create_diag_buffer(max_constants, value_size, batch_size),
        code_size=code_size
    )
    # print(sess.run(dsm.data_stack))
    # print(sess.run(dsm.inc_matrices(3, 0)))
    # print(sess.run(pretty_print_value(dsm.evaluate(sym.selected_word()))))
    # print(dsm.evaluate(sym.selected_word()))

    # print(sess.run(pretty_print_buffer(dsm.data_stack)))
    print(vocab.words[0])
    next_step = dsm.step()
    print(sess.run(pretty_print_buffer(next_step.data_stack)))
    print(sess.run(pretty_print_value(next_step.data_stack_pointer)))
    print(sess.run(pretty_print_value(next_step.pc)))

    # print(sess.run(next.pc))
    # print(sess.run(next.data_stack))
    # print(sess.run(next.pc))


if __name__ == "__main__":
    main()
