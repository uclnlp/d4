import copy
import d4.intermediate as im


# Define symbolic accessors
def data_stack_elem(index=None):
    """Symbolic accessor for the data stack"""
    return 'D', index


def return_stack_elem(index=None):
    """Symbolic accessor for the return stack"""
    return 'R', index


def fetch(addr=None, heap=None):
    """Symbolic accessor for the fetch command"""
    return '@', addr, heap


def store(value=None, addr=None, heap=None):
    """Symbolic accessor for the store command"""
    return '!', value, addr, heap


def heap():
    """Symbolic accessor for the heap"""
    return 'H',


def pc():
    """Symbolic accessor for the program counter"""
    return 'PC',


def convert_data_to_return(input=None):
    return "DATA_TO_RETURN", input


def convert_pc_to_return(input=None):
    return "PC_TO_RETURN", input


def convert_return_to_pc(input=None):
    return "RETURN_TO_PC", input


def convert_return_to_data(input=None):
    return "RETURN_TO_DATA", input


def next_pc(batch_to_positions=None):
    return 'NEXT_PC', batch_to_positions


def inc(value=None, amount=None, dim=None):
    return 'INC', value, amount, dim


# def tf_node(node=None):
#     return "TF_NODE", node


def inc_return_value(value=None, amount=None):
    return 'INC_RETURN_VALUE', value, amount


def data_stack_pointer():
    return 'DP',


def data_stack():
    return 'D',


def return_stack():
    return 'R',


def write_buffer(buffer=None, addr=None, value=None):
    return 'WRITE_BUFFER', buffer, addr, value


def return_stack_pointer():
    return 'RP',


def dec(value=None, amount=None, dim=None):
    return inc(value, -amount, dim)


def eq(arg1=None, arg2=None):
    return '=', arg1, arg2


def zero():
    return '0',


def one():
    return '1'


def gt(arg1=None, arg2=None):
    return '>', arg1, arg2


def vector_plus(arg1=None, arg2=None):
    return 'v+', arg1, arg2


def vector_minus(arg1=None, arg2=None):
    return 'v-', arg1, arg2


def vector_times(arg1=None, arg2=None):
    return 'v*', arg1, arg2


def vector_divide(arg1=None, arg2=None):
    return 'v/', arg1, arg2


def one_hot_add(arg1=None, arg2=None):
    return '+', arg1, arg2


def one_hot_sub(arg1=None, arg2=None):
    return '-', arg1, arg2


def one_hot_mul(arg1=None, arg2=None):
    return '*', arg1, arg2


def one_hot_div(arg1=None, arg2=None):
    return '/', arg1, arg2


def label(l=None):
    return 'L', l


def selected_word():
    return 'SEL_WORD',


def choice(pred=None, if_true=None, if_false=None):
    return 'CHOICE', pred, if_true, if_false


def constant(n=None):
    return 'CONSTANT', n


class SymbolicDSM:
    """
    Symbolic DSM
    """
    def __init__(self, stack_size, rstack_size, heap_size, value_size):
        # memory sizes
        self.value_size = value_size
        self.stack_size = stack_size
        self.rstack_size = rstack_size
        self.heap_size = heap_size
        # DSTACK
        self.data_stack = [data_stack_elem(self.data_stack_index(i))
                           for i in range(0, stack_size)]
        self.data_stack_top = 0
        # RSTACK
        self.return_stack = [return_stack_elem(self.return_stack_index(i))
                             for i in range(0, rstack_size)]
        self.return_stack_top = 0

        self.heap = heap()
        self.pc = pc()
        self.min_data_depth = 0
        self.min_return_depth = 0
        self.indices_in_batches = None

        # intermediate language word to symbolic command
        self.word_map = {
            im.dup(): self.dup,
            im.drop(): self.drop,
            im.swap(): self.swap,
            im.over(): self.over,
            im.inc(): self.inc,
            im.dec(): self.dec,
            im.step(): self.step,
            im.halt(): self.halt,
            im.fetch(): self.fetch,
            im.store(): self.store,
            im.exit(): self.exit,
            im.inc(): self.inc,
            im.dec(): self.dec,
            im.eq(): self.eq,
            im.gt(): self.gt,
            im.lt(): self.lt,
            im.drop(): self.drop,
            im.nop(): self.nop,
            im.init_do_loop(): self.init_do_loop,
            im.terminate_do_loop(): self.clean_do_loop,
            im.to_r(): self.to_r,
            im.r_from(): self.r_from,
            im.r_fetch(): self.r_fetch,
            im.up(): lambda: self.inc_data(1),
            im.vector_minus(): self.vector_minus,
            im.vector_plus(): self.vector_plus,
            im.vector_times(): self.vector_times,
            im.vector_divide(): self.vector_divide,
            im.one_hot_add(): self.one_hot_add,
            im.one_hot_sub(): self.one_hot_sub,
            im.one_hot_mul(): self.one_hot_mul,
            im.one_hot_div(): self.one_hot_div,
        }

    def halt(self):
        return self

    def copy(self):
        return copy.deepcopy(self)

    def inc_data(self, amount=1):
        self.data_stack_top += amount
        self.min_data_depth = min(self.min_data_depth, self.data_stack_top)

    def inc_return(self, amount=1):
        self.return_stack_top += amount
        self.min_return_depth = min(self.min_return_depth, self.return_stack_top)

    def data_stack_index(self, index):
        return index if index < self.stack_size / 2 else index - self.stack_size

    def return_stack_index(self, index):
        return index if index < self.rstack_size / 2 else index - self.rstack_size

    def top(self):
        return self.data_stack[self.data_stack_top]

    def next(self):
        return self.data_stack[self.data_stack_top - 1]

    def dup(self):
        self.inc_data()
        self.data_stack[self.data_stack_top] = self.data_stack[self.data_stack_top - 1]
        return self

    def over(self):
        self.inc_data()
        self.data_stack[self.data_stack_top] = self.data_stack[self.data_stack_top - 2]
        return self

    def swap(self):
        tmp = self.top()
        self.data_stack[self.data_stack_top] = self.next()
        self.data_stack[self.data_stack_top - 1] = tmp
        self.min_data_depth = min(self.min_data_depth, self.data_stack_top - 1)
        return self

    def inc(self):
        self.data_stack[self.data_stack_top] = inc(self.data_stack[self.data_stack_top], 1,
                                                   self.value_size)
        return self

    def dec(self):
        self.data_stack[self.data_stack_top] = inc(self.data_stack[self.data_stack_top], -1,
                                                   self.value_size)
        return self

    def eq(self):
        arg1 = self.top()
        arg2 = self.next()
        self.inc_data(-1)
        self.data_stack[self.data_stack_top] = eq(arg1, arg2)
        return self

    def gt(self):
        arg1 = self.top()
        arg2 = self.next()
        self.inc_data(-1)
        self.data_stack[self.data_stack_top] = gt(arg2, arg1)
        return self

    def lt(self):
        arg1 = self.top()
        arg2 = self.next()
        self.inc_data(-1)
        self.data_stack[self.data_stack_top] = gt(arg1, arg2)
        return self

    def binary_op(self, func):
        arg1 = self.top()
        arg2 = self.next()
        self.inc_data(-1)
        self.data_stack[self.data_stack_top] = func(arg2, arg1)

    def vector_plus(self):
        self.binary_op(vector_plus)

    def vector_minus(self):
        self.binary_op(vector_minus)

    def vector_times(self):
        self.binary_op(vector_times)

    def vector_divide(self):
        self.binary_op(vector_divide)

    def one_hot_add(self):
        self.binary_op(one_hot_add)

    def one_hot_sub(self):
        self.binary_op(one_hot_sub)

    def one_hot_mul(self):
        self.binary_op(one_hot_mul)

    def one_hot_div(self):
        self.binary_op(one_hot_div)

    def r_fetch(self):
        self.inc_data()
        self.data_stack[self.data_stack_top] = convert_return_to_data(
            self.return_stack[self.return_stack_top])
        return self

    def to_r(self):
        self.inc_return()
        self.return_stack[self.return_stack_top] = convert_data_to_return(self.top())
        self.inc_data(-1)
        return self

    def r_from(self):
        self.inc_data(1)
        self.data_stack[self.data_stack_top] = convert_return_to_data(
            self.return_stack[self.return_stack_top])
        self.inc_return(-1)
        return self

    def drop(self):
        self.inc_data(-1)

    def nop(self):
        pass

    def call(self, l):
        self.inc_return(1)
        self.return_stack[self.return_stack_top] = convert_pc_to_return(self.next_pc())
        self.pc = label(l)
        return self

    def exit(self):
        self.pc = convert_return_to_pc(self.return_stack[self.return_stack_top])
        self.inc_return(-1)
        return self

    def branch0(self, l):
        self.pc = choice(self.top(), self.next_pc(), label(l))
        self.inc_data(-1)
        return self

    def init_do_loop(self):
        self.inc_return(2)
        self.return_stack[self.return_stack_top] = convert_data_to_return(self.top())
        self.return_stack[self.return_stack_top - 1] = convert_data_to_return(self.next())
        self.inc_data(-2)
        return self

    def inc_do_loop(self, loop_start_label):
        index = self.return_stack[self.return_stack_top]
        limit = self.return_stack[self.return_stack_top - 1]
        self.return_stack[self.return_stack_top] = inc_return_value(index, 1)
        reached = eq(self.return_stack[self.return_stack_top], limit)
        self.pc = choice(reached, self.next_pc(), label(loop_start_label))

    def clean_do_loop(self):
        self.inc_return(-2)
        return self

    def branch(self, l):
        self.pc = label(l)
        return self

    def store(self):
        addr = self.top()
        value = self.next()
        self.heap = store(value, addr, self.heap)
        self.data_stack_top -= 2
        return self

    def fetch(self):
        self.data_stack[self.data_stack_top] = fetch(self.top(), self.heap)
        return self

    def constant(self, n):
        self.inc_data()
        self.data_stack[self.data_stack_top] = constant(n)
        return self

    def one(self):
        self.inc_data()
        self.data_stack[self.data_stack_top] = one()
        return self

    def zero(self):
        self.inc_data()
        self.data_stack[self.data_stack_top] = zero()
        return self

    def next_pc(self):
        return next_pc(self.indices_in_batches)

    def step(self):
        self.pc = self.next_pc()
        return self

    def execute_single_word(self, im_code):
        self.word_map[im_code]()
        return self

    def set_positions_in_batches(self, indices_in_batches):
        with_tuples = [(key, tuple(value)) for key, value in indices_in_batches.items()]
        self.indices_in_batches = frozenset(with_tuples)

    def execute_im(self, im_code):
        """
        Execute intermediate code on the symbolic DSM.

        :param im_code: the code to execute.
        """
        if im_code[0] == im.macro()[0]:
            for word in im_code[1]:
                self.execute_im(word)
        elif im_code[0].startswith('BRANCH'):
            label = im_code[1]
            if im_code[0] == im.branch()[0]:
                self.branch(label)
            else:
                self.branch0(label)
        elif im_code[0] == im.call()[0]:
            label = im_code[1]
            self.call(label)
        elif im_code[0] == im.inc_do_loop()[0]:
            label = im_code[1]
            self.inc_do_loop(label)
        elif im_code[0] == im.constant()[0]:
            self.constant(im_code[1])
        else:
            self.execute_single_word(im_code)

    def __str__(self):
        data_stack_changes = ["Slot {}:{}".format(self.data_stack_index(slot), value)
                              for slot, value in enumerate(self.data_stack)
                              if value != data_stack_elem(self.data_stack_index(slot))]
        return_stack_changes = ["Slot {}:{}".format(self.return_stack_index(slot), value)
                                for slot, value in enumerate(self.return_stack)
                                if value != return_stack_elem(self.return_stack_index(slot))]
        return ("(D: {}, [{},{}], R: {}, [{},{}], H: {}, PC: {})"
                .format(" ".join(data_stack_changes),
                        self.min_data_depth,
                        self.data_stack_top,
                        " ".join(return_stack_changes),
                        self.min_return_depth,
                        self.return_stack_top,
                        self.heap, self.pc)
                )


if __name__ == "__main__":
    dsm = SymbolicDSM(5, 5, 10, 10)
    dsm.constant(3)
    dsm.inc()
    dsm.constant(1)
    print('DSM:', dsm)
    dsm.dec()
    dsm.constant(2)
    print('DSM:', dsm)
    dsm.drop()
    dsm.dup()
    print('DSM:', dsm)

    print('DSTACK: ')
    for i, elem in enumerate(dsm.data_stack):
        print('*' if i == dsm.data_stack_top else ' ', elem)

    print('RSTACK: ')
    for i, elem in enumerate(dsm.return_stack):
        print('*' if i == dsm.return_stack_top else ' ', elem)

    print('min data depth: ', dsm.min_data_depth)
    print('HEAP: ', dsm.heap)
    print('PC: ', dsm.pc)
