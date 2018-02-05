import logging
import d4.interpreter as si
import tensorflow as tf
import numpy as np
from d4.dsm.loss import L2Loss


logging.basicConfig(level=logging.DEBUG)


def test_learn_to_do_addition_calls():
    batch_size = 1
    interpreter = si.SimpleInterpreter(30, 20, 15, batch_size, parallel_branches=True, do_normalise_pointer=False)
    for batch in range(0, batch_size):
        interpreter.load_code(": 2+ 1+ 1+ ; { choose 1+ 2+ } { choose 1+ 2+ } ", batch)
    trace = interpreter.execute(20)
    loss = L2Loss(trace[-1], interpreter)
    l2_loss = loss.l2_loss  # / batch_size

    opt = tf.train.AdamOptimizer(learning_rate=1.)
    opt_op = opt.minimize(l2_loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # presampled = [np.random.randint(0, 10) for i in range(0, 10)]
    for epoch in range(0, 1):
        for i in range(0, batch_size):
            sampled = np.random.randint(0, 10)
            interpreter.load_stack([sampled], i)
            loss.load_target_stack([sampled + 3], i)
        current_loss, _ = sess.run([l2_loss, opt_op], loss.current_feed_dict())
        print(current_loss)

if __name__ == "__main__":
    test_learn_to_do_addition_calls()
