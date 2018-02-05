import logging

import tensorflow as tf

import d4.dsm.extensible_dsm as edsm
import d4.interpreter as si
from d4.dsm.loss import L2Loss

logging.basicConfig(level=logging.DEBUG)


def test_learn_to_swap():
    batch_size = 1
    interpreter = si.SimpleInterpreter(3, 20, 15, batch_size,
                                       parallel_branches=True, do_normalise_pointer=True)
    for batch in range(0, batch_size):
        interpreter.load_code(">R { permute D0 R0 } R>",
                              batch)
    trace = interpreter.execute(5)
    state = trace[-1]
    loss = L2Loss(state, interpreter)
    l2_loss = loss.l2_loss / batch_size

    opt = tf.train.AdamOptimizer(learning_rate=0.1)
    opt_op = opt.minimize(l2_loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # presampled = [np.random.randint(0, 10) for i in range(0, 10)]
    xs = [i for i in range(0, 10)]
    ys = [i for i in range(0, 10)]

    for epoch in range(0, 100):
        # for i in range(0, batch_size):
        #     x = xs[i]
        #     y = ys[i]
        # random.shuffle(xs)
        for i in range(0, 10):
            # print("-" * 50)
            # print(i)
            # random.shuffle(ys)
            for j in range(0, batch_size):
                # x = xs[i]
                #     y = ys[i]
                x = xs[i]
                y = ys[j]
                interpreter.load_stack([x, y], j)
                loss.load_target_stack([y, x], j)
            current_loss, data_stack, data_stack_pointer, return_stack, return_stack_pointer, _ = sess.run(
                [l2_loss,
                 edsm.pretty_print_buffer(state.data_stack), edsm.pretty_print_value(state.data_stack_pointer),
                 edsm.pretty_print_buffer(state.return_stack), edsm.pretty_print_value(state.return_stack_pointer),
                 opt_op], loss.current_feed_dict())
            print("I")
            print(i)
            print("Diff")
            print(sess.run(edsm.pretty_print_buffer(loss.data_stack_diff), loss.current_feed_dict()))
            print(sess.run(edsm.pretty_print_value(loss.data_stack_pointer_diff), loss.current_feed_dict()))
            print("D")
            print(data_stack)
            print("DP")
            print(data_stack_pointer)
            print("R")
            print(return_stack)
            print("DP")
            print(return_stack_pointer)
            print("Loss")
            print(current_loss)
            # sess.run([state.data_stack], loss.current_feed_dict())


if __name__ == "__main__":
    test_learn_to_swap()
