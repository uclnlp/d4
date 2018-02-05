import tensorflow as tf
import d4.interpreter as si
import numpy as np


rand = np.random
rand.seed(1337)


def test_halt():
    bubble = """
    : BUBBLE
        DUP IF >R
            OVER OVER < IF SWAP THEN
            R> SWAP >R 1- BUBBLE R>
        ELSE
            DROP
        THEN
    ;

    : SORT
        1- DUP 0 DO >R R@ BUBBLE R> LOOP DROP
    ;
    SORT
    """

    max_length = 5

    interpreter = si.SimpleInterpreter(stack_size=2 * max_length + 10,
                                       value_size=2 * max_length + 1,
                                       min_return_width=25,
                                       batch_size=1,
                                       parallel_branches=True,
                                       collapse_forth=True,
                                       test_time_stack_size=2 * max_length + 10)

    interpreter.load_code(bubble, 0)
    sess = tf.Session()

    print("max steps", 10 * max_length * (max_length + 15) + 5)

    input_seq = rand.randint(2, 11, max_length)
    input_seq = np.append(input_seq, len(input_seq))

    interpreter.test_time_load_stack(input_seq, 0)

    test_trace, (next_state, step) = \
        interpreter.execute_test_time(sess,
                                      10 * max_length * (max_length + 15) + 5,
                                      use_argmax_pointers=False,
                                      use_argmax_stacks=False,
                                      test_halt=True)

    print(step)
    assert step < 200

if __name__ == "__main__":
    test_halt()
