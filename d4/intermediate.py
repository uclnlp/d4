"""
Intermediate code representation, without IF THEN etc, only using branch, call, etc.
(first tuple elem always used to check the command)

Intermediate code is divided into the following groups:

literals:
    - NUM, CONSTANT
DSTACK commands:
    - DUP, OVER, SWAP, DROP, 1+, 1-, UP
comparison operators:
    - <, >, =
HEAP commands:
    - @, !
control flow commands:
    - LABEL, CALL, BRANCH, BRANCH0, DO, LOOP, TERMINATE_DO, EXIT, HALT, STEP, NOP
RSTACK commands:
    - >R, R>, R@
vector commands:
    - V+, V-, V*, V/
macros:
    - MACRO, PIPELINE, PARALLEL
slot commands:
    - SLOT, CHOOSE, SAMPLE

"""


# literals


def number(n):
    return 'NUM', n


def constant(c=None):
    return 'CONSTANT', c


# DSTACK commands


def dup():
    return 'DUP',


def over():
    return 'OVER',


def swap():
    return 'SWAP',


def drop():
    return 'DROP',


def dec():
    return '1-',


def inc():
    return '1+',


def up():
    return 'UP',


# comparison operators


def eq():
    return '=',


def gt():
    return '>',


def lt():
    return '<',


# HEAP commands


def fetch():
    return '@',


def store():
    return '!',


# control flow commands


def label(l=None):
    return 'LABEL', l


def call(label_=None):
    return 'CALL', label_


def branch0(label_=None):
    return 'BRANCH0', label_


def branch(label_=None):
    return 'BRANCH', label_


def init_do_loop():
    return 'DO',


def inc_do_loop(label_=None):
    return 'LOOP', label_


def terminate_do_loop():
    return "TERMINATE_DO",


def exit():
    return 'EXIT',


def halt():
    return 'HALT',


def step():
    return 'STEP',


def nop():
    return 'NOP',


# RSTACK commands


def to_r():
    return '>R',


def r_from():
    return 'R>',


def r_fetch():
    return 'R@',


# vector commands


def vector_plus():
    return 'V+',


def vector_minus():
    return 'V-',


def vector_times():
    return 'V*',


def vector_divide():
    return 'V/',


# one-hot arithmetic ops


def one_hot_add():
    return '+',


def one_hot_sub():
    return '-',


def one_hot_mul():
    return '*',


def one_hot_div():
    return '/',


# macros


def macro(commands=None):
    return 'MACRO', commands


def pipeline(commands=None):
    return 'PIPELINE', commands


def parallel(left=None, right=None):
    ret_left = tuple(left) if left is not None else None
    ret_right = tuple(right) if right is not None else None
    return 'PARALLEL', ret_left, ret_right


# slot commands

def slot(decoder=None, encoder=None, id_=None, transformations=None):
    return 'SLOT', decoder, encoder, id_, transformations


def choose(choices=None):
    return 'CHOOSE', choices


def sample(choices=None):
    return 'SAMPLE', choices
