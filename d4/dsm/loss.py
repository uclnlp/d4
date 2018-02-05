import abc
import numpy as np
import tensorflow as tf
import d4.dsm.extensible_dsm as edsm


class DSMLoss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, dsm, interpreter):
        with tf.name_scope("loss"):
            self.dsm = dsm
            self.interpreter = interpreter

            data_stack_shape = [dsm.value_size, dsm.stack_size, dsm.batch_size]

            # DSTACK values
            # mask placeholder (for focusing on the active part of the stack)
            self.data_stack_mask = np.zeros(data_stack_shape)
            self.data_stack_mask_placeholder = tf.placeholder(tf.float32, data_stack_shape,
                                                              name="D_mask_pl")
            # target placeholder
            self.data_stack_target = np.zeros(data_stack_shape)
            self.data_stack_target_placeholder = tf.placeholder(tf.float32, data_stack_shape,
                                                                name="D_target_pl")

            # DSTACK pointer
            data_stack_pointer_shape = [dsm.stack_size, dsm.batch_size]
            self.data_stack_pointer_target = np.zeros(data_stack_pointer_shape)
            self.data_stack_pointer_target_placeholder = tf.placeholder(tf.float32,
                                                                        data_stack_pointer_shape,
                                                                        name="Dp_target_pl")
            # default stack pointer is at the top position
            self.data_stack_pointer_target[-1, :] = 1.0

            # PC
            pc_target_shape = [dsm.code_size, dsm.batch_size]
            self.pc_target_placeholder = tf.placeholder(tf.float32, pc_target_shape,
                                                        name="pc_target_pl")
            self.pc_target = np.zeros(pc_target_shape)

            # TODO: this may be different for different batches.
            #       Get the word index of the final line of code (code_size)
            # self.pc_target[-1,:] = 1.0

            # mask * (output - target)

            # DSTACK
            with tf.name_scope("D_diff"):
                self.data_stack_diff = self.data_stack_mask_placeholder \
                                       * (self.dsm.data_stack - self.data_stack_target_placeholder)

            with tf.name_scope("Dp_diff"):
                self.data_stack_pointer_diff = (self.dsm.data_stack_pointer
                                                - self.data_stack_pointer_target_placeholder)

            # DSTACK (pointer only)
            # the return stack should be empty
            with tf.name_scope("Rp_diff"):
                self.return_stack_diff = (self.dsm.return_stack_pointer
                                          - edsm.create_pointer(dsm.stack_size,
                                                                dsm.batch_size,
                                                                dsm.stack_size - 1))

            # position of the halting command (the last command)
            # # TODO in case of a shorter program inside a long batch, this might never be accessed
            # with tf.name_scope("halt_pc"):
            #     self.halt_pc = edsm.create_pointer(dsm.code_size, dsm.batch_size,
            #                                        dsm.code_size - 1)
            #
            # # program counter should end at the position of the (TODO first) halt command
            # with tf.name_scope("pc_diff"):
            #     self.pc_diff = self.dsm.pc - self.halt_pc

        pass

    @abc.abstractmethod
    def current_feed_dict(self, external_feed_dict=None):
        pass

    @abc.abstractmethod
    def load_target_stack(self, values, batch=0):
        """
        Loads target stack values per batch. Takes into account only those values,
        everything else is ignored (masked)

        :param values: target stack values
        :param batch: which batch
        """
        # data stack mask, ones in rows with values
        self.data_stack_mask[:, :len(values), batch] = 1.0
        self.data_stack_mask[:, len(values):, batch] = 0.0

        # target data stack, set its values to `values`
        self.data_stack_target[:, :, batch] = 0.0
        for row, value in enumerate(values):
            if isinstance(value, (int, np.int64)):
                self.data_stack_target[value, row, batch] = 1.0
            elif isinstance(value, float):
                self.data_stack_target[0, row, batch] = value
            else:
                print('[load_target_stack] Cannot recognise value type', type(value))
        # set stack pointer according to `values` size
        self.data_stack_pointer_target[:, batch] = 0.0
        self.data_stack_pointer_target[len(values) - 1, batch] = 1.0

        # print("Targets for batch:", batch)
        # print("Values", values)
        # print("Stack target")
        # print(self.data_stack_target[:, :, batch])
        # print("Stack mask")
        # print(self.data_stack_mask[:, :, batch])
        # print("Stack pointer")
        # print(self.data_stack_pointer_target[:, batch])


class CrossEntropyLoss(DSMLoss):
    """
    Cross entropy loss on DSM
    """
    def __init__(self, dsm: edsm.DSMState, interpreter, use_logits=False):
        """
        """
        with tf.name_scope("ce_loss"):
            super().__init__(dsm, interpreter)

            self.values_lengths = np.zeros([dsm.batch_size])
            self.values_lengths_placeholder = tf.placeholder(tf.int32, [dsm.batch_size])
            self.data_stack_values = np.zeros([dsm.stack_size, dsm.batch_size])
            self.data_stack_values_placeholder = tf.placeholder(tf.int32,
                                                                [dsm.stack_size, dsm.batch_size])

            def cross_entropy_no_logits(prediction, target):
                eps = 1e-10
                return -tf.reduce_sum(target * tf.log(tf.maximum(prediction, eps)), 1)

            # DSTACK
            # pointer
            self.data_stack_pointer_loss = cross_entropy_no_logits(
                tf.transpose(self.dsm.data_stack_pointer),
                tf.transpose(self.data_stack_pointer_target_placeholder))

            if use_logits:
                self.data_stack_loss = cross_entropy_no_logits(
                    tf.nn.softmax(tf.transpose(tf.reshape(
                        self.data_stack_mask_placeholder * self.dsm.data_stack,
                        shape=[dsm.value_size, -1]))),
                    tf.nn.softmax(tf.transpose(tf.reshape(
                        self.data_stack_mask_placeholder * self.data_stack_target_placeholder,
                        shape=[dsm.value_size, -1])))
                )
            else:
                self.data_stack_loss = cross_entropy_no_logits(
                    tf.reshape(self.data_stack_mask_placeholder * self.dsm.data_stack,
                               shape=[dsm.value_size, -1]),
                    tf.reshape((self.data_stack_mask_placeholder
                                * self.data_stack_target_placeholder),
                               shape=[dsm.value_size, -1])
                )

            self.loss = (tf.reduce_sum(self.data_stack_pointer_loss)
                         + tf.reduce_sum(self.data_stack_loss))

            # # TODO loss on the return stack ?
            # # TODO loss on the heap ?

    def current_feed_dict(self, external_feed_dict=None):
        """
        Builds up the feed_dict for the loss on top of the interpreter's feed_dict

        :return: feed_dict for the computational graph
        """
        result = self.interpreter.current_feed_dict(external_feed_dict)
        result[self.data_stack_mask_placeholder] = self.data_stack_mask
        result[self.data_stack_pointer_target_placeholder] = self.data_stack_pointer_target
        result[self.data_stack_target_placeholder] = self.data_stack_target
        result[self.pc_target_placeholder] = self.pc_target
        result[self.values_lengths_placeholder] = self.values_lengths
        result[self.data_stack_values_placeholder] = self.data_stack_values
        return result

    def load_target_stack(self, values, batch=0):
        super().load_target_stack(values, batch)


class L2Loss(DSMLoss):
    """
    Convenience class to simplify losses on DSM states.
    """

    def __init__(self, dsm: edsm.DSMState, interpreter, regularisation_weight=0.001):
        """
        Build Loss for given DSM state assumed to have been produced by the given interpreter.
        :param dsm: the state to define the loss on.
        :param interpreter: the interpreter. Needs to provide the feed dict that grounds the dsm.
        :regularisation_weight: weight for the L2 regularisation of slot variables
        """
        with tf.name_scope("l2_loss"):
            super().__init__(dsm, interpreter)

            # TODO heap loss
            # self.heap_loss = ...

            # TODO unused
            # PC
            # self.pc_loss = tf.nn.l2_loss(self.pc_diff, name="pc")

            # DSTACK
            self.data_stack_loss = tf.nn.l2_loss(self.data_stack_diff, name="D")
            self.data_stack_pointer_loss = tf.nn.l2_loss(self.data_stack_pointer_diff, name="Dp")

            # RSTACK
            self.return_stack_pointer_loss = tf.nn.l2_loss(self.return_stack_diff)

            # TODO shove params here
            self.l2_loss = self.data_stack_loss + self.data_stack_pointer_loss
            # + self.return_stack_pointer_loss

            # variable regularisation
            with tf.name_scope("slot_variables"):
                self.slot_variables = [tf.reshape(v, [-1])
                                       for v in tf.trainable_variables()
                                       if v.name.startswith("slots")]

            self.l2_regulariser = 0.0
            if len(self.slot_variables) > 0:
                self.l2_regulariser = tf.nn.l2_loss(tf.concat(0, self.slot_variables),
                                                    name="l2_regulariser")

            # wonky
            self.regularised_l2_loss = self.l2_loss + regularisation_weight * self.l2_regulariser

            self.loss = self.l2_loss

            # self.loss = self.l2_loss + regularisation_weight *

    def current_feed_dict(self, external_feed_dict=None):
        """
        Builds up the feed_dict for the loss on top of the interpreter's feed_dict

        :return: feed_dict for the computational graph
        """
        result = self.interpreter.current_feed_dict(external_feed_dict)
        result[self.data_stack_mask_placeholder] = self.data_stack_mask
        result[self.data_stack_pointer_target_placeholder] = self.data_stack_pointer_target
        result[self.data_stack_target_placeholder] = self.data_stack_target
        result[self.pc_target_placeholder] = self.pc_target
        return result

    def load_target_stack(self, values, batch=0):
        super().load_target_stack(values, batch)
