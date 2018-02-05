import logging
import d4.interpreter as si
import tensorflow as tf
import numpy as np
from d4.dsm.loss import L2Loss


def test_learn_to_add_three():
    logging.basicConfig(level=logging.DEBUG)
    batch_size = 10
    interpreter = si.SimpleInterpreter(10, 15, 15, batch_size, parallel_branches=True)
    for batch in range(0, batch_size):
        interpreter.load_code(": 2+ 1+ 1+ ; { choose 1+ 2+ } { choose 1+ 2+ } ", batch)
    trace = interpreter.execute(10)
    loss = L2Loss(trace[-1], interpreter)
    l2_loss = loss.l2_loss  # / batch_size

    opt = tf.train.AdamOptimizer(learning_rate=0.1)
    opt_op = opt.minimize(l2_loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # presampled = [np.random.randint(0, 10) for i in range(0, 10)]

    current_loss = 0

    for epoch in range(0, 100):
        for i in range(0, batch_size):
            sampled = np.random.randint(0, 10)
            interpreter.load_stack([sampled], i)
            loss.load_target_stack([sampled + 3], i)
        current_loss, _ = sess.run([l2_loss, opt_op], loss.current_feed_dict())
        current_loss /= batch_size
        print(epoch, current_loss)

    assert current_loss < 0.47

if __name__ == "__main__":
    test_learn_to_add_three()
