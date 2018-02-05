import logging

import tensorflow as tf

import d4.dsm.extensible_dsm as edsm
import d4.interpreter as si

eps = 0.00001
code = """
VARIABLE LENGTH
VARIABLE QUESTION
CREATE REPR_BUFFER 4 ALLOT
CREATE NUM_BUFFER 4 ALLOT

VARIABLE REPR
VARIABLE NUM

REPR_BUFFER REPR !
NUM_BUFFER NUM !

3 LENGTH !

MACRO: STEP
  REPR @ 1+ REPR !
  NUM @ 1+ NUM !
;

MACRO: CURRENT_NUM NUM @ @ ;
MACRO: CURRENT_REPR REPR @ @ ;

QUESTION @ >R

\ the main loop iterating over the numbers and their representations
LENGTH @ 0 DO
  CURRENT_REPR >R CURRENT_NUM     \ putting representation on R to condition on it R: [Q REPR_i]
  { observe R0 R-1 ->
    choose  NOP DROP v+ v- }      \ operating the number
  R> DROP                         \ clean up
  STEP
LOOP

\ conclude by reading the question and processing the current state one last time
{ observe R0 -> choose v+ v- NOP DROP }

\ clean up R
R> DROP



"""


def test_algebra():
    sess = tf.Session()
    logging.basicConfig(level=logging.DEBUG)
    interpreter = si.SimpleInterpreter(10, 20, 5, 1)
    print("Yo")
    interpreter.load_code(code)
    interpreter.load_stack([])
    interpreter.load_heap([3, 2,  # length, question
                           1, 2, 4, 3,  # representations
                           1.0, 2.0, 2.0, 1.0  # numbers
                           ])
    interpreter.create_initial_dsm()
    print("Initialising")
    sess.run(tf.initialize_all_variables())
    trace, _ = interpreter.execute_test_time(sess, 30)
    state = trace[-1]
    edsm.print_dsm_state_np(data_stack=state[interpreter.test_time_data_stack],
                            data_stack_pointer=state[interpreter.test_time_data_stack_pointer],
                            pc=state[interpreter.test_time_pc],
                            heap=state[interpreter.test_time_heap], interpreter=interpreter)


def gold_algebra(question, numbers, representantions):
    stack = [0.0]
    for number, reprs in zip(numbers, representantions):
        combined = (question + reprs) % 4
        stack.append(number)
        if combined == 0:
            pass
        elif combined == 1:
            del stack[-1]
        elif combined == 2:
            stack[-2] = stack[-1] + stack[-2]
            del stack[-1]
        elif combined == 3:
            stack[-2] = stack[-1] - stack[-2]
            del stack[-1]
    if question == 0:
        return stack[-1]
    elif question == 1:
        return -stack[-1]


if __name__ == "__main__":
    print(gold_algebra(0, [1.0, 2.0, 3.0], [1, 2, 2]))
    test_algebra()
