import logging

from tests.evaluate_specs import evaluate_specs


def test_digit_plus():
    digit_plus = """
    : digit+
    DUP 1 = IF >R
        DUP 9 = IF
            DROP 0 R> ELSE 1+ R> 1-
        THEN
    THEN
    >R
      BEGIN DUP WHILE
        1- SWAP DUP 9 = IF
          R> 1+ >R
          DROP 0
        ELSE
          1+ SWAP
        THEN
      REPEAT
      DROP
      R>
    ;

    digit+
"""
    specs = [
        ([1, 2, 1], [4, 0], digit_plus),
        ([9, 0, 1], [0, 1], digit_plus),
        ([9, 1, 0], [0, 1], digit_plus),
        ([1, 8, 1], [0, 1], digit_plus),
        ([9, 9, 1], [9, 1], digit_plus),
        ([5, 5, 1], [1, 1], digit_plus),

    ]
    evaluate_specs(5, 10, 5, 50, specs)


def test_add_digits():
    add_digits = """
    : digit+
    DUP 1 = IF >R
        DUP 9 = IF
            DROP 0 R> ELSE 1+ R> 1-
        THEN
    THEN
    >R
      BEGIN DUP WHILE
        1- SWAP DUP 9 = IF
          R> 1+ >R
          DROP 0
        ELSE
          1+ SWAP
        THEN
      REPEAT
      DROP
      R>
    ;

    : add-digits ( a1 b1 a2 b2 ... an bn carry n -- r1 r2 ... r_{n+1} )
        DUP 0 = IF
            DROP
        ELSE
            >R
            digit+
            SWAP R> 1- SWAP >R
            add-digits
            R>
        THEN
    ;

    add-digits
"""
    specs = [
        ([1, 2, 1, 4, 0, 2], [0, 3, 5], add_digits),
        ([5, 5, 4, 5, 1, 2], [1, 1, 0], add_digits),
        ([9, 0, 9, 1, 0, 2], [1, 0, 0], add_digits),
    ]
    evaluate_specs(10, 10, 10, 80, specs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_digit_plus()
    test_add_digits()
