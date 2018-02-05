import d4.compiler as sc
import logging


def test_slots():
    result = sc.compile("{ observe D-1 D0 -> manipulate D-1 }")
    assert result == [('SLOT',
                       ('MANIPULATE', (('D', -1),)),
                       ('OBSERVE', (('D', -1), ('D', 0))), 'L0', ())]

    result = sc.compile("{ static -> manipulate D-1 }")
    assert result == [('SLOT',
                       ('MANIPULATE', (('D', -1),)),
                       ('STATIC', 'None'), 'L0', ())]

    result = sc.compile("{ static -> choose 1+ 1- }")
    assert result == [('SLOT',
                       ('CHOOSE', (('1+',), ('1-',))),
                       ('STATIC', 'None'), 'L0', ())]

    result = sc.compile("{ manipulate D-1 D0 }")
    assert result == [('SLOT',
                       ('MANIPULATE', (('D', -1), ('D', 0))),
                       ('STATIC', 'None'), 'L0', None)]

    result = sc.compile("{ observe D0 -> tanh -> linear 5 -> sigmoid -> choose 0 1 } ")
    assert result == [('SLOT',
                       ('CHOOSE', (('CONSTANT', 0), ('CONSTANT', 1))),
                       ('OBSERVE', (('D', 0),)),
                       'L0', ('tanh', ('linear', 5), 'sigmoid'))]


def test_constants():
    result = sc.compile("1 2 3")
    assert result == [('CONSTANT', 1), ('CONSTANT', 2), ('CONSTANT', 3)]


def test_constant_def():
    result = sc.compile("0 CONSTANT TRUE TRUE")
    assert result == [('CONSTANT', 0)]


def test_macro_def():
    result = sc.compile("macro: 2DUP OVER OVER ; IF 2DUP THEN")
    assert result == [('BRANCH0', 'L0'), ('OVER',), ('OVER',), ('LABEL', 'L0')]


def test_create_def():
    result = sc.compile("CREATE BUFFER 4 ALLOT VARIABLE X BUFFER X")
    assert result == [('CONSTANT', 0), ('CONSTANT', 4)]


def test_longer():
    code = """
    0 CONSTANT FALSE
    1 CONSTANT TRUE
    VARIABLE X
    VARIABLE Y
    VARIABLE SUCCESS

    : UNIFY_CONSTANT ( constant addr -- )
      DUP @ IF
        @ = SUCCESS !
      ELSE
        !
      THEN
    ;
    TRUE SUCCESS !
    3 X1 !
    3 X1 UNIFY_CONSTANT
    SUCCESS @
    """
    result = sc.compile(code)
    print(result)


def test_variable_def():
    result = sc.compile("VARIABLE X VARIABLE Y X Y")
    assert result == [('CONSTANT', 0), ('CONSTANT', 1)]

    result = sc.compile("VARIABLE X VARIABLE Y IF Y THEN")
    print(result)
    # assert result == [('BRANCH0', 'L0'), ('CONSTANT', 1), ('LABEL', 'L0')]
    result = sc.compile("VARIABLE X VARIABLE Y IF Y IF X THEN THEN")
    print(result)


def test_comments():
    result = sc.compile("1 2 3 \\ weeee, look at me, I'm a comment!!")
    assert result == [('CONSTANT', 1), ('CONSTANT', 2), ('CONSTANT', 3)]

    result = sc.compile("""
        1+ ( im a comment too!! )
        4 5 \\ mee too!
        butimnot
        """)
    assert result == [('1+',), ('CONSTANT', 4), ('CONSTANT', 5), ('CALL', 'butimnot')]


def test_core_word():
    result = sc.compile("1+")
    assert result == [('1+',)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_constant_def()
    test_comments()
    test_constants()
    test_core_word()
    test_create_def()
    test_longer()
    test_macro_def()
    test_slots()
    test_variable_def()
