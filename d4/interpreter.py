import numpy as np
import tensorflow as tf
import logging

from collections import defaultdict

from d4.dsm.loss import DSMLoss
import d4.compiler as compiler
import d4.dsm.extensible_dsm as edsm
from d4.dsm.extensible_dsm import symbol_match
import d4.dsm.symbolic_dsm as sym
import d4.intermediate as im


def remove_labels(compiled):
    """
    Remove labels from code, and create mapping from a label it to code line position.

    For example, code:
        ('MACRO', (('BRANCH', 1),))
        ('LABEL', 0)
        ('MACRO', (('1+',), ('1+',), ('EXIT',)))
    returns:
        ('MACRO', (('BRANCH', 1),))
        ('MACRO', (('1+',), ('1+',), ('EXIT',)))
    and:
        {0: 1}

    Args:
        compiled: compiled code

    Returns:
        code without labels
        mapped label positions {label ID -> code line}

    """
    label_positions = {}
    without_labels = []
    for word in compiled:
        if word[0] == im.label()[0]:
            label = word[1]  # label points to the command after it
            label_positions[label] = len(without_labels)
        else:
            without_labels.append(word)
    return without_labels, label_positions


def replace_labels(code):
    """
    Replaces free labels in `code` with integers, and accordingly any label call / branch.

    For example, code:
        ('BRANCH', 'L0')
        ('LABEL', '2+')
        ('1+',)
        ('EXIT',)
        ('LABEL', 'L0')
    returns:
        ('BRANCH', 1)
        ('LABEL', 0)
        ('1+',)
        ('EXIT',)
        ('LABEL', 1)
    and:
        {'2+': 0, 'L0': 1}

    Args:
        code: the input IM code.

    Returns:
        code with labels replaced by numbers
        a {label -> ID} dictionary

    """
    labels = {}

    def find_labels(words):
        """ Find all the labels in the code """
        for word in words:
            if word[0] == im.label()[0]:
                label = word[1]
                labels[label] = len(labels)

    find_labels(code)
    label_ops = {im.branch()[0], im.branch0()[0], im.call()[0], im.inc_do_loop()[0]}

    def replace(words):
        """ Recursively replace labels in the code """
        result = []
        for word in words:
            if word[0] in label_ops:
                result.append((word[0], labels[word[1]]))
            elif word[0] == im.label()[0]:
                result.append(im.label(labels[word[1]]))
            elif symbol_match(word, im.parallel()):
                left, right = word[1:]
                result.append(im.parallel(replace(left), replace(right)))
            elif symbol_match(word, im.pipeline()):
                words = word[1]
                replaced_words = (replace(w) for w in words)
                result.append(im.pipeline(replaced_words))
            elif symbol_match(word, im.slot()):
                dec, enc, id_, trans = word[1:]
                if symbol_match(dec, im.choose()) or symbol_match(dec, im.sample()):
                    result.append(im.slot((dec[0], replace(dec[1])), enc, id_, trans))
                else:
                    result.append(word)
            else:
                result.append(word)
        return tuple(result)

    replaced = replace(code)

    return replaced, labels


def replace_constants(compiled):
    """
    Replace values of constants with their generated integer IDs.

    For example, code:
        ('CONSTANT', 12)
        ('DROP',)
    results in:
        ('CONSTANT', 0)
        ('DROP',)
    and:
        {12: 0}

    Args:
        compiled: compiled code

    Returns:
        code with constant values replaced by IDs
        the mapping {constant value -> ID}

    """
    constants = {}

    def replace(words):
        """ Recursively replace constant values to IDs in the code"""
        result = []
        for word in words:
            if word[0] == im.constant()[0]:
                if word[1] not in constants:
                    constants[word[1]] = len(constants)
                result.append(im.constant(constants[word[1]]))
            elif symbol_match(word, im.parallel()):
                left, right = word[1:]
                result.append(im.parallel(replace(left), replace(right)))
            elif symbol_match(word, im.pipeline()):
                words = word[1]
                replaced_words = (replace(w) for w in words)
                result.append(im.pipeline(replaced_words))
            elif symbol_match(word, im.slot()):
                dec, enc, id_, trans = word[1:]
                if symbol_match(dec, im.choose()) or symbol_match(dec, im.sample()):
                    result.append(im.slot((dec[0], replace(dec[1])), enc, id_, trans))
                else:
                    result.append(word)
            else:
                result.append(word)
        return tuple(result)

    replaced = replace(compiled)

    return replaced, constants


def inject_steps(im_code):
    """
    Injects DSM step transitions where needed to make sure the PC always progresses.

    For example, the code:
        ('MACRO', (('BRANCH', 1),))
        ('MACRO', (('1+',), ('1+',), ('EXIT',)))
        ('MACRO', (('>',), ('CONSTANT', 0)))
        ('PARALLEL', (('MACRO', (('DROP',), ('DROP',))),), (('MACRO', (('DROP',),)),))
    results in:
        ('MACRO', (('BRANCH', 1),))
        ('MACRO', (('1+',), ('1+',), ('EXIT',)))
        ('MACRO', (('>',), ('CONSTANT', 0), ('STEP',)))
        ('PIPELINE', (('PARALLEL',
                       (('MACRO', (('DROP',), ('DROP',))),),
                       (('MACRO', (('DROP',),)),)), ('STEP',)))

    Args:
        im_code: the code to inject to

    Returns:
        a list of words such that every word is guaranteed to progress.

    """
    result = []
    macro_terminators = {im.step()[0], im.branch()[0], im.branch0()[0], im.call()[0],
                         im.exit()[0], im.inc_do_loop()[0], im.slot()[0]}
    for word in im_code:
        if word[0] == im.macro()[0]:
            final_arg_typ = word[1][-1][0]
            if final_arg_typ not in macro_terminators:
                result.append(im.macro(word[1] + (im.step(),)))
            else:
                result.append(word)
        elif symbol_match(word, im.parallel()):
            result.append(im.pipeline((word, im.step())))
        elif (word[0] == im.slot()[0]
              and (word[1][0] == im.choose()[0] or word[1][0] == im.sample()[0])):
            dec, enc, label, transformations = word[1:]
            choices = dec[1]
            injected = [choice if choice[0] in macro_terminators else im.macro((choice, im.step()))
                        for choice in choices]
            result.append(im.slot((dec[0], tuple(injected)), enc, label, transformations))
        elif word[0] not in macro_terminators:
            result.append(im.macro((word, im.step())))
        else:
            result.append(word)

    return result


def merge_pipelines(code):
    """
    Merges commands into a pipeline

    TODO example

    Args:
        code: intermediate code

    Returns:
        code with non-branched commands merged in pipelines
    """
    result = []
    current = None
    seq_types = {im.pipeline()[0], im.macro()[0]}
    branch_types = {im.branch()[0], im.branch0()[0], im.call()[0], im.inc_do_loop()[0]}

    def get_last_step(im_code):
        if im_code[0] in seq_types:
            return get_last_step(im_code[1][-1])
        else:
            return im_code

    for word in code:
        if current is not None and current[0] in seq_types:
            last_step = get_last_step(current)
            if last_step[0] not in branch_types:
                current = im.pipeline((current, word))
            else:
                result.append(current)
                current = word
        else:
            if current is not None:
                result.append(current)
            current = word
    if current is not None:
        result.append(current)
    return result


class SimpleInterpreter:
    """
    Interprets (batched) forth sketches.
    """

    def __init__(self,
                 stack_size,
                 value_size,
                 min_return_width,
                 batch_size=1,
                 parallel_branches=True,
                 do_normalise_pointer=True,
                 test_time_stack_size=None,
                 collapse_forth=True,
                 merge_pipelines_=False,
                 init_weight_stddev=1.0):

        self.merge_pipelines = merge_pipelines_
        self.collapse_forth = collapse_forth
        self.parallel_branches = parallel_branches
        self.heap_size = value_size
        self.value_size = value_size
        self.stack_size = stack_size
        self.batch_size = batch_size
        self.min_return_width = min_return_width

        with tf.name_scope("interpreter_init"):
            with tf.name_scope("code_elements"):
                self.code = defaultdict(str)
                self.code_placeholder = tf.placeholder(
                    tf.float32, (None, None, batch_size), name="CODE")
                self.code_value = None

                self.constants_placeholder = tf.placeholder(
                    tf.float32, (value_size, None, batch_size), name="CONSTANTS")
                self.constants_value = None

                self.labels_placeholder = tf.placeholder(
                    tf.float32, (None, None, batch_size), name="LABELS")
                self.labels_value = None
            with tf.name_scope("train"):
                self.init_data_stack_placeholder = tf.placeholder(
                    tf.float32, (value_size, stack_size, batch_size), name="D")
                self.init_data_stack_value = np.zeros([value_size, stack_size, batch_size])

                self.init_data_stack_pointer_placeholder = tf.placeholder(
                    tf.float32, (stack_size, batch_size), name="DP")
                # TOS initialised to the last element
                self.init_data_stack_pointer_value = np.zeros([stack_size, batch_size])
                self.init_data_stack_pointer_value[-1, :] = 1.0

                self.init_heap_placeholder = tf.placeholder(
                    tf.float32, (value_size, self.heap_size, batch_size), name="H")
                self.init_heap_value = np.zeros([value_size, self.heap_size, batch_size])
                # initialise heap to zero values
                self.init_heap_value[0, :, :] = 1.0

            # test-time variables
            with tf.name_scope("test"):
                if test_time_stack_size is None:
                    self.test_time_stack_size = stack_size
                else:
                    self.test_time_stack_size = test_time_stack_size
                # data stack
                self.test_time_data_stack = tf.placeholder(
                    tf.float32, (value_size, self.test_time_stack_size, batch_size), name="D")
                self.test_time_data_stack_pointer = tf.placeholder(
                    tf.float32, (self.test_time_stack_size, batch_size), name="DP")
                # return stack
                self.test_time_return_stack = tf.placeholder(
                    tf.float32, (None, self.test_time_stack_size, batch_size), name="R")
                self.test_time_return_stack_pointer = tf.placeholder(
                    tf.float32, (self.test_time_stack_size, batch_size), name="RP")

                # heap
                self.test_time_heap = tf.placeholder(
                    tf.float32, (value_size, self.heap_size, batch_size), name="H")
                # PC
                self.test_time_pc = tf.placeholder(tf.float32, (None, batch_size), name="PC")
                # test time inits
                self.test_time_init_data_stack_value = np.zeros(
                    [value_size, self.test_time_stack_size, batch_size])
                # self.test_time_init_data_stack_pointer_placeholder = tf.placeholder(
                #     tf.float32, (self.test_time_stack_size, batch_size), name="test_time_init_DP")
                self.test_time_init_data_stack_pointer_value = np.zeros(
                    [self.test_time_stack_size, batch_size])
                self.test_time_init_data_stack_pointer_value[-1, :] = 1.0

        self.init_dsm = None
        self.test_time_dsm = None
        self.execution_trace = []
        self.requires_update = True
        self.slots = {}
        self.vocab = None
        self.final_code = None
        self.words = None  # mapping from word names to functions
        self.return_width = None
        self.code_size = None
        self.sample_decoders = None
        self._external_heap = False

        self.init_weight_stddev = init_weight_stddev

        self.do_normalise_pointer = do_normalise_pointer

        # self._create_initial_dsm()

    def load_code(self, code, batch=0):
        """
        Loads the code to a given batch index

        Args:
            code: the Forth code string to load.
            batch: the batch to load it into

        Returns:
            nothing

        """
        self.code[batch] = code
        self.requires_update = True

    # TODO: doesn't support floats
    def load_stack(self, values, batch=0, last_float=False):
        """
        Loads the stack of a batch.

        :param values: the list of values to put on the stack.
        :param batch: the batch number.
        """
        self.init_data_stack_value[:, :, batch] = 0.0
        self.init_data_stack_pointer_value[:, batch] = 0.0

        input_len = len(values)
        for row, value in enumerate(values):
            if last_float and row == input_len - 1:
                self.init_data_stack_value[0, row, batch] = value
            self.init_data_stack_value[value, row, batch] = 1.0

        self.init_data_stack_pointer_value[len(values) - 1, batch] = 1.0

    # TODO: doesn't support floats
    def test_time_load_stack(self, values, batch=0, last_float=False):
        """
        Loads the test time stack of a batch.

        :param values: the list of values to put on the stack.
        :param batch: the batch number.
        """
        self.test_time_init_data_stack_value[:, :, batch] = 0.0
        self.test_time_init_data_stack_pointer_value[:, batch] = 0.0
        input_len = len(values)
        for row, value in enumerate(values):
            if last_float and row == input_len - 1:
                self.test_time_init_data_stack_value[0, row, batch] = value
            self.test_time_init_data_stack_value[value, row, batch] = 1.0

        self.test_time_init_data_stack_pointer_value[len(values) - 1, batch] = 1.0

    def load_heap(self, values, batch=0):
        """
        Loads values onto the heap.

        :param values: the list of values to put on the stack. Skips empty-list elements.
        :param batch: the batch number. If None, values is expected to be the full Tensor.
        """

        if batch is None:
            self.init_heap_value = values
        else:
            self.init_heap_value[:, 0:len(values), batch] = 0.0
            for row, value in enumerate(values):
                if isinstance(value, int):
                    self.init_heap_value[value, row, batch] = 1.0
                elif isinstance(value, float):
                    self.init_heap_value[0, row, batch] = value

    def set_heap(self, tensor):
        """
        Sets heap externally.

        :param tensor: a [value_size x heap_size x batch_size] tensor representing heap values
        """
        self._external_heap = True
        self.init_heap_placeholder = tensor

    def execute(self, num_steps):
        """
        Creates an execution RNN of `num_steps` length.

        :param num_steps: number of steps.
        :return: the list of DSMState objects at each time step.
        """
        if self.requires_update:
            logging.info("Interpreter is updated")
            self.create_initial_dsm()
        current_steps = len(self.execution_trace)
        for i in range(current_steps - 1, num_steps):
            self.execution_trace.append(self.execution_trace[-1].step())
        return self.execution_trace

    def execute_test_time(self, sess, max_steps, use_argmax_pointers=False, use_argmax_stacks=False,
                          test_halt=False, debug=False, external_feed_dict=None,
                          save_only_last_step=False):
        """
        Executes the program(s) at test time. This means that no RNN graph is built to be trained
        through back-propagation. Instead a single DSM transition graph is created, and its
        placeholders are fed with the output of the last run.

        Args:
            sess: tensorflow session to use.
            max_steps: maximum number of steps to perform. We could potentially check for
            convergence to a stationary point, and terminate earlier.
            use_argmax_pointers: should pointers be argmaxed (it helps to execute a single command).
            use_argmax_stacks: should stack values be argmaxed (it helps to discretise the stack).
            test_halt: should we terminate if all batches reached the halt state.
            debug: guess!
            external_feed_dict: pass through external components to the feed dict (external models)
            save_only_last_step: save only the last step's output, otherwise save all steps

        Returns:
            list of feed_dict which stores the results of the transitions and assigns them to the
                interpreter's test_X placeholders.
            (next_state, step) next state dsm
        """
        if self.requires_update:
            logging.info("Interpreter is updated")
            self.create_initial_dsm()
        # pass init values into test_time_dsm
        feed_dict = {
            self.test_time_data_stack: self.test_time_init_data_stack_value,
            self.test_time_data_stack_pointer: self.test_time_init_data_stack_pointer_value,
            self.test_time_return_stack: edsm.create_buffer(self.test_time_stack_size,
                                                            self.return_width,
                                                            self.batch_size,
                                                            as_tf=False),
            self.test_time_return_stack_pointer: edsm.create_pointer(self.test_time_stack_size,
                                                                     self.batch_size, -1,
                                                                     as_tf=False),
            self.test_time_heap: self.init_heap_value,
            self.test_time_pc: edsm.create_pointer(self.code_size,
                                                   self.batch_size, 0,
                                                   as_tf=False),
            self.code_placeholder: self.code_value,
            self.constants_placeholder: self.constants_value,
            self.labels_placeholder: self.labels_value,
        }

        if external_feed_dict is not None:
            feed_dict.update(external_feed_dict)

        import copy
        trace = [copy.copy(feed_dict)]
        next_state = self.next_test_state  # self.test_time_dsm.step()

        def discretise_pointer(pointer):
            """ Discretise the pointer (used in test mode) """
            if not use_argmax_pointers:
                return pointer
            result = np.zeros(np.shape(pointer))
            argmax = np.argmax(pointer, 0)
            for col, row in zip(range(0, result.shape[1]), argmax):
                result[row, col] = 1.0
            return result

        def discretise_stack(stack):
            """ Discretise the stack (used in test mode) """
            if not use_argmax_stacks:
                return stack
            result = np.zeros(np.shape(stack))
            argmax = np.argmax(stack, 0)
            i1, i2 = np.indices(np.shape(result)[1:])
            result[argmax, i1, i2] = 1.0
            return result

        # iterate until max_steps or convergence: read out result, pass in again.
        step = 0
        halt = False
        while step < max_steps and not halt:
            data_stack, data_stack_pointer, return_stack, return_stack_pointer, heap, pc = (
                sess.run([next_state.data_stack, next_state.data_stack_pointer,
                          next_state.return_stack, next_state.return_stack_pointer,
                          next_state.heap, next_state.pc], feed_dict)
            )

            feed_dict[self.test_time_data_stack] = discretise_stack(data_stack)
            feed_dict[self.test_time_data_stack_pointer] = discretise_pointer(data_stack_pointer)

            feed_dict[self.test_time_return_stack] = discretise_stack(return_stack)
            feed_dict[self.test_time_return_stack_pointer] = discretise_pointer(
                return_stack_pointer)

            # TODO: sharpen heap
            feed_dict[self.test_time_heap] = heap

            feed_dict[self.test_time_pc] = discretise_pointer(pc)

            if not save_only_last_step:
                trace.append(copy.copy(feed_dict))

            step += 1

            # check whether ALL batches have reached the halt state.
            if test_halt:
                halt = True
                for batch in range(0, self.batch_size):
                    halt = halt and feed_dict[self.test_time_pc][-1, batch] > 0.9

            if debug:
                print("Step {}".format(step))
                edsm.print_dsm_state_np(data_stack, data_stack_pointer, return_stack,
                                        return_stack_pointer, pc=pc, interpreter=self)
        # print(step)
        if save_only_last_step:
            trace.append(copy.copy(feed_dict))

        return trace, (next_state, step)

    def current_feed_dict(self, external_feed_dict=None):
        """
        Returns a feed dict that captures the state of the interpreter.
        :param external_feed_dict: add the external feed dict
        :return: mostly initial values to the DSM state.
        """

        feed_dict = {
            self.code_placeholder: self.code_value,
            self.constants_placeholder: self.constants_value,
            self.labels_placeholder: self.labels_value,
            self.init_data_stack_placeholder: self.init_data_stack_value,
            self.init_data_stack_pointer_placeholder: self.init_data_stack_pointer_value,
        }
        if not self._external_heap:
            feed_dict[self.init_heap_placeholder] = self.init_heap_value

        if external_feed_dict is not None:
            feed_dict.update(external_feed_dict)

        return feed_dict

    def produce_decoder(self, dec):
        """
        Produced the decoder of the encoder -> decoder slot, per specifications
        (CHOOSE, SAMPLE, MANIPULATE, PERMUTE).

        Args:
            dec: intermediate code line specifying the decoder

        Returns:
            DSMDecoder per code specification

        """
        # TODO cut this out nicely
        if symbol_match(dec, im.choose()) or symbol_match(dec, im.sample()):    # CHOOSE or SAMPLE
            with tf.name_scope("CHOOSE_or_SAMPLE"):
                choices = dec[1]
                assembled_choices = []
                for choice in choices:
                    sym_dsm = sym.SymbolicDSM(self.stack_size * 2,
                                              self.stack_size * 2,
                                              self.heap_size,
                                              self.value_size)
                    assembled = edsm.SymbolicDSMWord(sym_dsm, choice)
                    assembled_choices.append(assembled)
                if symbol_match(dec, im.choose()):      # CHOOSE
                    with tf.name_scope("CHOOSE"):
                        return edsm.ChoiceDecoder(assembled_choices)
                elif symbol_match(dec, im.sample()):    # SAMPLE
                    with tf.name_scope("SAMPLE"):
                        decoder = edsm.SampleDecoder(assembled_choices)
                        self.sample_decoders.append(decoder)
                        return decoder
        elif dec[0] == 'MANIPULATE':    # MANIPULATE
            return edsm.ManipulateDecoder(dec[1], self.value_size, self.return_width)
        elif dec[0] == 'PERMUTE':   # PERMUTE
            return edsm.PermuteDecoder(dec[1])
        else:
            raise KeyError(dec)

    def produce_linear_transformer(self, linear_in_dim, linear_out_dim):
        """
        Produces a linear layer transformation

        :param linear_in_dim: layer input size
        :param linear_out_dim: layer output size
        :return: DSMDecoder per code specification
        """
        with tf.name_scope("linear"):
            with tf.name_scope("weights"):
                # [linear_in_dim, linear_out_dim]
                weights = tf.Variable(tf.random_normal([linear_out_dim, linear_in_dim],
                                                       stddev=self.init_weight_stddev))
            with tf.name_scope("bias"):
                # [linear_out_dim, 1]
                bias = tf.Variable(tf.random_normal([linear_out_dim, 1],
                                                    stddev=self.init_weight_stddev))

            def transform(x):
                """ Linear transformation function """
                with tf.name_scope("LINEAR"):
                    # [linear_in_dim, batch_size]
                    return tf.matmul(weights, x) + bias

            return transform

    def produce_encoder(self, enc, output_dim, transformations):
        """
        Produced the encoder of the encoder -> decoder slot, per specifications
        (STATIC, OBSERVE, and hidden layer transformations).

        Args:
            enc: intermediate code line specifying the encoder
            output_dim: encoder output dimensionality
            transformations: transformations applied to the encoder (linear, sigmoid, tanh)

        Returns:
            DSMEncoder, per code specification

        """
        # build up a list of applied transformations
        dim = output_dim
        trans_functions = []
        for trans in reversed(transformations if transformations is not None else []):
            if isinstance(trans, tuple) and trans[0].lower() == "linear":
                linear_in_dim = trans[1]
                linear_out_dim = dim
                trans_functions.append(self.produce_linear_transformer(linear_in_dim,
                                                                       linear_out_dim))
                dim = linear_in_dim
            elif trans.lower() == 'sigmoid':
                trans_functions.append(lambda x: tf.sigmoid(x, name="SIGMOID"))
            elif trans.lower() == 'tanh':
                trans_functions.append(lambda x: tf.tanh(x, name="TANH"))

        # build the encoder
        if enc[0] == 'STATIC':
            with tf.name_scope("STATIC"):
                input_encoder = edsm.StaticEncoder(dim, self.init_weight_stddev)
        elif enc[0] == 'OBSERVE':
            with tf.name_scope("OBSERVE"):
                input_encoder = edsm.ObserveEncoder(enc[1], dim, self.value_size,
                                                    self.return_width, self.init_weight_stddev)
        else:
            raise KeyError(enc)

        def transformation(x):
            """ Helper function for applying transformations on the encoder """
            with tf.name_scope("transformations"):
                result = x
                for f in reversed(trans_functions):
                    result = f(result)
            return result

        # return the encoder with the applied tranformations
        return edsm.Transformer(input_encoder, transformation)

    def create_initial_dsm(self):
        """
        Compiles all the code, and initialises the DSM
        """
        words = {}
        word_ids = {}
        code_size = 0
        max_constants = 0
        max_labels = 0
        constant_dicts = [None] * self.batch_size
        code_dicts = [None] * self.batch_size
        label_dicts = [None] * self.batch_size
        label_positions_dict = [dict()] * self.batch_size
        constant_mapping_dict = [dict()] * self.batch_size
        code_int_sequences = [list()] * self.batch_size
        vocab = edsm.AssembledVocab()
        batch_to_labels = {}

        self.sample_decoders = []
        self.final_code = {}

        # batch compilation...
        for batch in range(0, self.batch_size):
            logging.debug("Batch: {}".format(batch))

            code = self.code[batch]
            logging.debug("Compiling...")
            flat = compiler.compile(code, self.parallel_branches)

            # replace original labels with integer indexed labels, remember mapping
            replaced_labels, labels = replace_labels(flat)
            logging.debug("Replaced Labels: {}".format(replaced_labels))
            logging.debug("Labels dictionary: {}".format(labels))

            # replace original constants with integer ids
            replaced_constants, constants = replace_constants(replaced_labels)
            logging.debug("Replaced Constants: {}".format(replaced_constants))
            logging.debug("Constants dictionary:: {}".format(constants))

            # collapse code into macros
            collapsed = (compiler.collapse_im(replaced_constants)
                         if self.collapse_forth else replaced_constants)
            logging.debug("Collapsed: {}".format(collapsed))

            # extract mapping from label ids to code positions
            removed_labels, label_positions = remove_labels(collapsed)
            logging.debug("Code without Labels: {}".format(removed_labels))
            logging.debug("Label Positions: {}".format(label_positions))

            # injects STEPs where necessary
            final_code = inject_steps(removed_labels) + [im.halt()]
            logging.debug("Steps Injected: {}".format(final_code))

            # merge pipelines?
            if self.merge_pipelines:
                final_code = merge_pipelines(final_code)
            self.final_code[batch] = final_code

            # final code logging
            logging.debug("Final code:")
            position_to_label = {v: k for k, v in label_positions.items()}
            for i, word in enumerate(final_code):
                logging.debug("{} {} : {}".format(i, position_to_label.get(i, " "), word))
            logging.debug("...DONE")

            # prepare dictionaries
            batch_to_labels[batch] = labels  # TODO: unused?
            constant_mapping_dict[batch] = constants
            label_positions_dict[batch] = label_positions

            code_size = max(code_size, len(final_code))
            max_constants = max(max_constants, len(constants))
            max_labels = max(max_labels, len(labels))

        self.return_width = max(self.min_return_width, code_size)

        # executes unique compiled words on the DSM
        for batch in range(0, self.batch_size):
            logging.debug("Batch: {}".format(batch))
            final_code = self.final_code[batch]

            # labels = batch_to_labels[batch]
            # batch_to_labels = batch_to_label_positions[batch]

            # execute a piece of IM code on the dsm. Resolve labels and constants.
            def register_word(word_, assembled):
                words[word_] = assembled
                word_ids[word_] = len(word_ids)
                vocab.add_word(assembled)
                logging.debug("Registered: {}".format(str(assembled)))

            code_int_sequences[batch] = []
            for position, word in enumerate(final_code):
                if word not in words:

                    def to_single_word(commands):
                        """ Encapsulates a list of commands in a pipeline """
                        if len(commands) == 1:
                            return commands[0]
                        else:
                            return im.pipeline(commands)

                    def create_word(input_word):
                        """ Turns a word into an assembled DSM word"""
                        if symbol_match(input_word, im.slot()):
                            with tf.variable_scope("slots"):
                                dec, enc, label, transformations = input_word[1:]
                                # produce decoder & encoder
                                with tf.name_scope("decoder"):
                                    produced_dec = self.produce_decoder(dec)
                                with tf.name_scope("encoder"):
                                    produced_enc = self.produce_encoder(enc,
                                                                        produced_dec.input_dim(),
                                                                        transformations)
                                with tf.name_scope("ENCODER_DECODER_WORD"):
                                    created_word = edsm.EncoderDecoderWord(produced_enc,
                                                                           produced_dec)
                                self.slots[label] = created_word
                        elif symbol_match(input_word, im.parallel()):
                            true, false = input_word[1:]
                            created_word = edsm.ParallelWord(create_word(to_single_word(true)),
                                                             create_word(to_single_word(false)))
                        elif symbol_match(input_word, im.pipeline()):
                            words = input_word[1]
                            steps = [create_word(s) for s in words]
                            created_word = edsm.PipelineWord(steps)
                        else:
                            sym_dsm = sym.SymbolicDSM(self.stack_size * 2, self.stack_size * 2,
                                                      self.heap_size, self.value_size)
                            created_word = edsm.SymbolicDSMWord(sym_dsm, input_word)
                        return created_word

                    logging.debug("Registering word {}".format(word))
                    register_word(word, create_word(word))

                code_int_sequences[batch].append(word_ids[word])
            logging.debug("Code ID sequence: {}".format(code_int_sequences[batch]))

        # Building up code/label/constant matrices
        for batch in range(self.batch_size):
            logging.debug("Batch: {}".format(batch))

            # first we instantiate the actual assembled words:
            code_dicts[batch] = np.zeros((len(vocab), code_size))
            for position, id in enumerate(code_int_sequences[batch]):
                code_dicts[batch][id, position] = 1.0
                vocab.words[id].append_code_position(batch, position)
            # fill up the rest with with halt
            for position in range(len(code_int_sequences[batch]), code_size):
                code_dicts[batch][word_ids[im.halt()], position] = 1.0
            # and now we have a code matrix
            logging.debug("Code Matrix:\n{}".format(code_dicts[batch]))

            # then we build a matrix of labels (row: id, column: code line)
            label_dicts[batch] = np.zeros((max_labels, code_size))
            for label, position in label_positions_dict[batch].items():
                label_dicts[batch][label, position] = 1.0

            logging.debug("Label Dict:\n{}".format(label_dicts[batch]))

            # and a matrix of constants
            constant_dicts[batch] = np.zeros((max_constants, self.value_size))
            for value, int_id in constant_mapping_dict[batch].items():
                if isinstance(value, int):
                    constant_dicts[batch][int_id, value] = 1.0
                elif isinstance(value, float):
                    constant_dicts[batch][int_id, 0] = value
            logging.debug("Constant Dict:\n{}".format(constant_dicts[batch]))

        # batching up the matrices
        packed_code = np.array(code_dicts)  # [batch_size, vocab_size, code_size]
        with tf.name_scope("code_value"):
            # [vocab_size, code_size, batch_size]
            self.code_value = np.transpose(packed_code, [1, 2, 0])

        packed_labels = np.array(label_dicts)  # [batch_size, max_labels, code_size]
        with tf.name_scope("labels_value"):
            self.labels_value = np.transpose(packed_labels, [2, 1, 0])

        packed_constants = np.array(constant_dicts)  # [batch_size, max_constants, value_size]
        with tf.name_scope("constants_value"):
            # [value_size, max_constants, batch_size]
            self.constants_value = np.transpose(packed_constants, [2, 1, 0])

        # vocabulary words
        logging.debug("Vocab size: {}".format(len(words)))
        logging.debug("Code size: {}".format(code_size))
        for i, word in enumerate(vocab.words):
            logging.debug("Word {}: {}".format(i, word))

        le_heap = self.init_heap_placeholder
        le_test_time_heap = self.test_time_heap

        # set the states...

        # used for training
        with tf.name_scope("train_dsm"):
            self.init_dsm = edsm.DSMState(
                code_size=code_size,
                data_stack=self.init_data_stack_placeholder,
                data_stack_pointer=self.init_data_stack_pointer_placeholder,
                return_stack=edsm.create_buffer(self.stack_size, self.return_width, self.batch_size),
                return_stack_pointer=edsm.create_pointer(self.stack_size, self.batch_size, -1),
                heap=le_heap,
                pc=edsm.create_pointer(code_size, self.batch_size, 0),
                code=self.code_placeholder,
                vocab=vocab,
                labels=self.labels_placeholder,
                constants=self.constants_placeholder,
                do_normalise_pointer=self.do_normalise_pointer
            )

        # used for execution at test time.
        with tf.name_scope("test_dsm"):
            self.test_time_dsm = edsm.DSMState(
                code_size=code_size,
                data_stack=self.test_time_data_stack,
                data_stack_pointer=self.test_time_data_stack_pointer,
                return_stack=self.test_time_return_stack,
                return_stack_pointer=self.test_time_return_stack_pointer,
                heap=le_test_time_heap,
                pc=self.test_time_pc,
                code=self.code_placeholder,
                vocab=vocab,
                labels=self.labels_placeholder,
                constants=self.constants_placeholder,
                return_width=self.return_width,
                do_normalise_pointer=self.do_normalise_pointer
            )

        self.execution_trace = [self.init_dsm]
        self.requires_update = False
        self.vocab = vocab
        self.words = words
        self.code_size = code_size
        # TODO: set choice words to sample
        # self.next_test_state = self.test_time_dsm.step("test_step", argmax_merger=True)
        self.next_test_state = self.test_time_dsm.step("test")


def main():
    logging.basicConfig(level=logging.DEBUG)
    stack_size = 6
    value_size = 5
    min_return_width = 5
    batch_size = 2
    # set the interpreter up
    interpreter = SimpleInterpreter(stack_size, value_size, min_return_width, batch_size,
                                    parallel_branches=True, merge_pipelines_=True)
    # fill batches with code and the initial stack values
    for batch in range(0, batch_size):
        interpreter.load_code(": 2+ 1+ 1+ ; 2 1 DROP SWAP 2+ ", batch)
        interpreter.load_stack([1, 2], batch)

    # execute the interpreter to get the trace

    trace = interpreter.execute(10)
    # loss = DSMLoss(trace[-1], interpreter)

    # do tensorflow stuff
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    print("Code")
    print(sess.run(edsm.pretty_print_buffer(trace[0].code),
                   feed_dict=interpreter.current_feed_dict()))
    for i, step in enumerate(trace[:]):
        print("-" * 50)
        print("Step " + str(i))
        print("Data Stack")
        print(sess.run(edsm.pretty_print_buffer(step.data_stack),
                       feed_dict=interpreter.current_feed_dict()))
        print("Data stack pointer")
        print(sess.run(edsm.pretty_print_value(step.data_stack_pointer),
                       feed_dict=interpreter.current_feed_dict()))

        print("Return Stack")
        print(sess.run(edsm.pretty_print_buffer(step.return_stack),
                       feed_dict=interpreter.current_feed_dict()))
        print("Return stack pointer")
        print(sess.run(edsm.pretty_print_value(step.return_stack_pointer),
                       feed_dict=interpreter.current_feed_dict()))
        print('.' * 10)

        print("PC")
        pc = sess.run(edsm.pretty_print_value(step.pc),
                      feed_dict=interpreter.current_feed_dict())
        print(pc)
        print("Selected Word")
        print(sess.run(edsm.pretty_print_value(step.evaluate(sym.selected_word())),
                       feed_dict=interpreter.current_feed_dict()))

        for batch in range(0, interpreter.batch_size):
            print("Batch {}".format(batch))
            for word_index in range(0, step.code_size):
                score = pc[batch, word_index]
                # print(score)
                if score > 0.1:
                    print("{} {}".format(score, interpreter.words[interpreter.final_code[batch][word_index]]))

                    # print("HEAP")
                    # print(sess.run(edsm.pretty_print_buffer(step.heap), feed_dict=interpreter.current_feed_dict()))

                    # print(edsm.pretty_print_value(trace[-2].evaluate(sym.selected_word())))

    loss = DSMLoss(trace[-1], interpreter)
    for batch in range(0, batch_size):
        loss.load_target_stack([1], batch)

    print(sess.run([edsm.pretty_print_buffer(loss.data_stack_diff),
                    edsm.pretty_print_value(loss.data_stack_pointer_diff)], feed_dict=loss.current_feed_dict()))

    # for variable in tf.all_variables():
    #     if variable.name.startswith('slots'):
    #         print(variable.name)

    #         print(interpreter.slots['L0'])


if __name__ == "__main__":
    main()
