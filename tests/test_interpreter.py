import logging

import tensorflow as tf

import d4.dsm.extensible_dsm as edsm
import d4.interpreter as si

from tests.evaluate_specs import evaluate_specs


def test_arithmetics():
    """ INC DEC + - * / """
    specs_inc = [
        ([2], [3], "1+"),
        ([0], [1], "1+"),
        ([4], [0], "1+"),   # cyclic inc
    ]
    specs_dec = [
        ([1], [0], "1-"),
        ([2], [1], "1-"),
        ([0], [4], "1-"),  # cyclic dec
    ]
    specs_plus = [
        ([1, 2], [3], "+"),
        ([0, 1], [1], "+"),
        ([3, 0], [3], "+"),
        ([0, 0], [0], "+"),
        ([2, 3], [0], "+"),  # cyclic plus
        ([4, 2], [1], "+"),  # cyclic plus
    ]
    specs_minus = [
        ([2, 1], [1], "-"),
        ([2, 0], [2], "-"),
        ([4, 2], [2], "-"),
        ([4, 4], [0], "-"),
        ([1, 2], [4], "-"),  # cyclic minus
        ([0, 1], [4], "-"),
    ]
    specs_times = [
        ([1, 2], [2], "*"),
        ([0, 2], [0], "*"),  # multiplication by zero
        ([2, 0], [0], "*"),  # multiplication by zero
        ([2, 2], [4], "*"),
        ([3, 2], [1], "*"),  # cyclic times
        ([3, 3], [4], "*"),  # cyclic times
    ]
    specs_div = [
        ([2, 1], [2], "/"),
        ([2, 2], [1], "/"),
        ([4, 2], [2], "/"),
        ([4, 1], [4], "/"),
        ([3, 2], [1], "/"),

        ([0, 1], [0], "/"),  # dividing zero
        ([0, 2], [0], "/"),  # dividing zero
        ([0, 4], [0], "/"),  # dividing zero

        ([0, 0], [4], "/"),  # division by zero
        ([1, 0], [4], "/"),  # division by zero
        ([2, 0], [4], "/"),  # division by zero
        ([3, 0], [4], "/"),  # division by zero
        ([4, 0], [4], "/"),  # division by zero
    ]

    full_specs = {
        '1+': specs_inc,
        '1-': specs_dec,
        '+': specs_plus,
        '-': specs_minus,
        '*': specs_times,
        '/': specs_div
    }

    for op, spec in full_specs.items():
        print('Testing: ', op)
        evaluate_specs(5, 5, 5, 3, spec)


def test_dstack_commands():
    """ DUP SWAP OVER DROP NOP """

    specs_nop = [
        ([1, 2], [1, 2], "NOP"),
        ([], [], "NOP"),
        ([0, 0, 0, 0], [0, 0, 0, 0], "NOP"),
    ]
    specs_dup = [
        ([1], [1, 1], "DUP"),
        ([1, 4, 2], [1, 4, 2, 2], "DUP"),
        ([0], [0, 0], "DUP"),
        ([0, 4], [0, 4, 4], "DUP"),
    ]
    specs_swap = [
        ([2, 3], [3, 2], "SWAP"),
        ([0, 0], [0, 0], "SWAP"),
        ([1, 2], [2, 1], "SWAP"),
        ([1, 2, 3], [1, 3, 2], "SWAP"),
        ([1, 2, 3], [1, 2, 3], "SWAP SWAP"),
    ]
    specs_over = [
        ([1, 2], [1, 2, 1], "OVER"),
        ([], [1, 3, 1], "1 3 OVER"),
        ([1, 3], [1, 3, 1, 3], "OVER OVER"),
    ]
    specs_drop = [
        ([1, 2, 3], [1, 2], "DROP"),
        ([4, 0], [4], "DROP"),
        ([1], [], "DROP"),
        ([0], [], "DROP"),
    ]

    full_specs = {
        'NOP': specs_nop,
        'DROP': specs_drop,
        'SWAP': specs_swap,
        'DUP': specs_dup,
        'OVER': specs_over,
    }

    for op, spec in full_specs.items():
        print('Testing: ', op)
        evaluate_specs(5, 5, 5, 5, spec)


def test_literals():
    """ LITERALS """
    specs = [
        ([], [1], "1"),
        ([], [0, 4, 1, 2], "0 4 1 2"),
        ([], [2, 0, 1, 3, 4], "2 0 1 3 4"),
        ([], [0, 0, 0, 0, 0], "0 0 0 0 0"),

    ]
    print("Testing: LITERALS")
    evaluate_specs(5, 5, 5, 2, specs)


def test_call():
    """ function definition and call """
    specs = [
        ([], [2], ": 2+ 1+ 1+ ; 0 2+"),
        ([2], [0], ": 2- 1- 1- ; 2-"),
    ]
    print("Testing: function call")
    evaluate_specs(5, 5, 5, 3, specs)


def test_from_to_r():
    """ >R R> """
    specs = [
        ([1, 3], [1], ">R"),
        ([1, 3], [], ">R >R"),
        ([1, 3], [1, 3], ">R R>"),
        ([1, 3], [1, 3], ">R >R R> R>"),
        ([1, 3], [2, 3], ">R 1+ R>"),
    ]
    print("Testing: >R R>")
    evaluate_specs(5, 5, 5, 5, specs)


def test_store_fetch():
    """ ! @ """
    specs = [
        ([], [1], "1 2 ! 2 @"),
        ([], [], "1 2 !"),
        ([], [0], "2 @"),  # empty mem needs to be initiated with zeros
    ]
    print("Testing: ! @")
    evaluate_specs(5, 5, 5, 5, specs)


def test_comparison():
    """ > = """
    specs = [
        ([2, 1], [1], ">"),  # true
        ([2, 2], [0], ">"),  # false
        ([1, 2], [0], ">"),  # false
        ([1, 0], [1], ">"),  # true
        ([0, 1], [0], ">"),  # false
        ([0, 0], [1], "="),  # true
        ([1, 1], [1], "="),  # true
        ([4, 4], [1], "="),  # true
        ([0, 1], [0], "="),  # false
        ([1, 0], [0], "="),  # false
        ([3, 4], [0], "="),  # false
        ([1, 1], [0], "> IF 1 ELSE 0 THEN"),
        ([2, 1], [1], "> IF 1 ELSE 0 THEN"),
        ([2, 3], [0], "> IF 1 ELSE 0 THEN"),
        ([2, 2], [0], "> IF 1 ELSE 0 THEN"),
        ([2, 1], [0], "= IF 1 ELSE 0 THEN"),
        ([2, 2], [1], "= IF 1 ELSE 0 THEN"),
        ([2, 3], [0], "= IF 1 ELSE 0 THEN"),
    ]
    print("Testing: > =")
    evaluate_specs(5, 5, 5, 5, specs)


def test_if_then():
    """ IF..ELSE..THEN """
    specs = [
        ([1], [1, 2], "IF 1 THEN 2"),
        ([0], [2], "IF 1 THEN 2"),
        ([1], [1, 3], "IF 1 ELSE 2 THEN 3"),
        ([0], [2, 3], "IF 1 ELSE 2 THEN 3")
    ]
    print("Testing: IF..ELSE..THEN")
    evaluate_specs(5, 5, 5, 3, specs)


def test_while_loop():
    """ BEGIN..WHILE..LOOP """
    specs = [
        ([1, 2], [3], "BEGIN DUP WHILE 1- SWAP 1+ SWAP REPEAT DROP"),
    ]
    print("Testing: BEGIN..WHILE..LOOP")
    evaluate_specs(5, 5, 5, 10, specs)


def test_do_loop():
    """ DO..LOOP """
    specs = [
        ([1], [3], "2 0 DO 1+ LOOP"),
        ([1], [3, 3], "2 0 DO 1+ LOOP DUP"),
    ]
    print("Testing: DO..LOOP")
    evaluate_specs(5, 5, 5, 10, specs)


def test_recursion():
    specs = [
        ([1, 2], [3], ": ADD DUP IF 1- SWAP 1+ SWAP ADD ELSE DROP THEN ; ADD"),
    ]
    print("Testing: ...recursion")
    evaluate_specs(5, 5, 5, 10, specs)


def test_seq_plus_one():
    code = """
    : seq+1
        DUP 0 > IF
            1- SWAP
            1+ >R seq+1 R>
        ELSE
            DROP
        THEN
    ;
    seq+1
        """
    specs = [
        ([4, 3, 2, 7, 8, 5], [5, 4, 3, 8, 9], code)
    ]
    print("Testing: ...plus one to a sequence")
    evaluate_specs(20, 10, 10, 30, specs)


def test_fibonacci():
    code = """
    : + BEGIN DUP WHILE 1- SWAP 1+ SWAP REPEAT DROP ;
    : FIBONACCI DUP 1 > IF 1- DUP 1- FIBONACCI SWAP FIBONACCI + THEN ;
    FIBONACCI
    """

    specs = [
        ([1], [1], code),
        ([2], [1], code),
        ([3], [2], code),
        ([4], [3], code),
        ([5], [5], code),
    ]
    print("Testing: ...fibonacci")
    evaluate_specs(20, 20, 20, 100, specs, parallel_branches=True)


def test_up():
    """
    Testing a special word UP that increases the stack depth by 1, but makes no guarantees
    what to put on the stack. This is useful as a preprocessing step for manipulative slots.
    """
    specs = [
        ([], [1], "1 UP DROP"),
    ]
    evaluate_specs(5, 5, 5, 5, specs)


def test_test_time_execution():
    sess = tf.Session()

    interpreter = si.SimpleInterpreter(5, 20, 5, 1,
                                       collapse_forth=True,
                                       parallel_branches=True,
                                       merge_pipelines_=True)
    # interpreter.load_code("1.0 v-")
    # interpreter.load_code("8 2 - 3 /")
    interpreter.load_code("8 2 -")
    # interpreter.test_time_load_stack([], last_float=False)
    trace, (_, steps) = interpreter.execute_test_time(sess, 10, use_argmax_pointers=True,
                                                      test_halt=True)
    state = trace[-1]
    edsm.print_dsm_state_np(data_stack=state[interpreter.test_time_data_stack],
                            data_stack_pointer=state[interpreter.test_time_data_stack_pointer],
                            pc=state[interpreter.test_time_pc],
                            interpreter=interpreter)
    print(len(trace))


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    # + - * / 1+ 1-
    test_arithmetics()

    # DUP SWAP OVER DROP NOP
    test_dstack_commands()

    # LITERALS
    test_literals()

    # function call
    test_call()

    # >R R>
    test_from_to_r()

    # ! @
    test_store_fetch()

    # = >
    test_comparison()

    # IF..ELSE..THEN
    test_if_then()

    # BEGIN..WHILE..LOOP
    test_while_loop()

    # DO..LOOP
    test_do_loop()

    # other tests
    test_recursion()
    test_seq_plus_one()
    test_fibonacci()

    # test_up()
    # test_test_time_execution()
