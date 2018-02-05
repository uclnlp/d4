import logging

from tests.evaluate_specs import evaluate_specs


def test_recursive_bubble():
    bubble = """
    : BUBBLE ( a1 ... an n-1 -- one-bubble-pass )
      DUP IF >R
        OVER OVER < IF SWAP THEN
        R> SWAP >R 1- BUBBLE R>
      ELSE
        DROP
      THEN
    ;
    BUBBLE
    """
    specs = [
        ([1, 2, 1], [2, 1], bubble),
        ([2, 1, 1], [2, 1], bubble),
        ([1, 2, 3, 2], [3, 1, 2], bubble),
        ([1, 3, 2, 2], [3, 1, 2], bubble),
        ([2, 6, 8, 2], [8, 2, 6], bubble),
        ([8, 6, 2, 2], [8, 6, 2], bubble)
    ]
    evaluate_specs(5, 10, 5, 20, specs, debug=False, parallel_branches=True)


def test_bubble_sort():
    bubble = """
    : BUBBLE ( a1 ... an n-1 -- one-bubble-pass )
      DUP IF >R
        OVER OVER < IF SWAP THEN
        R> SWAP >R 1- BUBBLE R>
      ELSE
        DROP
      THEN
    ;
    : SORT ( a1 ... an n -- sorted )
      1- DUP 0 DO >R R@ BUBBLE R> LOOP DROP
    ;
    SORT
    """
    specs = [
        ([1, 2, 3, 3], [3, 2, 1], bubble),
        ([1, 3, 2, 3], [3, 2, 1], bubble),
        ([1, 1, 2, 3], [2, 1, 1], bubble),
        ([1, 2, 3, 4, 4], [4, 3, 2, 1], bubble),
        ([4, 3, 2, 1, 4], [4, 3, 2, 1], bubble)
    ]
    evaluate_specs(20, 10, 10, 80, specs, False)


def ignore_test_bubble_sort_with_slots():
    bubble = """
    : BUBBLE ( a1 ... an n-1 -- one-bubble-pass )
      DUP IF >R
        { observe D0 D-1 -> choose NOP SWAP }
        R> SWAP >R 1- BUBBLE R>
      ELSE
        DROP
      THEN
    ;
    : SORT ( a1 ... an n -- sorted )
      1- DUP 0 DO >R R@ BUBBLE R> LOOP DROP
    ;
    SORT
    """
    specs = [
        ([1, 2, 3, 3], [3, 2, 1], bubble),
        ([3, 4, 3, 3], [4, 3, 3], bubble),
    ]
    evaluate_specs(10, 5, 10, 50, specs, True)


def ignore_test_loop_bubble_sort():
    # Address 0: length
    # Address 1 - length: values
    bubble = """
    : I 0 ;
    : N 1 ;
    : B 2 ;
    : DEC_I I @ 1- I ! ;
    : P_I I @;
    : P_I+1 I @ 1+;
    : A_I P_I @ ;
    : A_I+1 P_I+1 @ ;
    : SWAP-ELEM
      A_I A_I+1 2DUP > IF P_I ! P_I+1 ! ELSE 2DROP THEN
    ;
    : BUBBLE
      N @ I !
      BEGIN I @ WHILE SWAP-ELEM DEC_I @REPEAT
    ;
    : SORT ( -- )
      N @ 0 DO BUBBLE LOOP
    ;
    SORT
    """
    specs = [
        ([1, 2, 3, 3], [3, 2, 1], bubble),
        ([1, 3, 2, 3], [3, 2, 1], bubble),
        ([1, 1, 2, 3], [2, 1, 1], bubble),
        ([1, 2, 3, 4, 4], [4, 3, 2, 1], bubble),
    ]
    evaluate_specs(20, 10, 10, 80, specs, False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print('kvaav')
    test_recursive_bubble()
