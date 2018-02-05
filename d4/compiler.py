import re
import logging
from funcparserlib.parser import (some, a, many, with_forward_decls, skip)
from funcparserlib.lexer import (Token, make_tokenizer)

import d4.intermediate as im


# Well known functions / keywords
KEYWORDS = {':', ';', 'if', 'then', 'else', 'begin', 'while', 'repeat',
            'do', 'loop', '{', '}', 'choose', 'manipulate', 'observe', '->',
            'macro:', 'variable', 'allot', 'create', 'constant'}


# Core Forth words
CORE_WORDS = {
    # DSTACK ops
    'dup': im.dup(),
    'swap': im.swap(),
    'over': im.over(),
    'drop': im.drop(),
    '1+': im.inc(),
    '1-': im.dec(),
    'up': im.up(),
    # HEAP ops
    '@': im.fetch(),
    '!': im.store(),
    # comparison ops
    '=': im.eq(),
    '>': im.gt(),
    '<': im.lt(),
    # RSTACK ops
    '>r': im.to_r(),
    'r>': im.r_from(),
    'r@': im.r_fetch(),
    # vector-arithmetic ops
    'v+': im.vector_plus(),
    'v-': im.vector_minus(),
    'v*': im.vector_times(),
    'v/': im.vector_divide(),
    # one-hot arithmetic ops
    '+': im.one_hot_add(),
    '-': im.one_hot_sub(),
    '*': im.one_hot_mul(),
    '/': im.one_hot_div(),
    # other
    'nop': im.nop(),
    'abort': im.halt(),
}


# sort keywords to match longest words first
CORE_WORDS_SORTED = [i for i in CORE_WORDS.keys()]
CORE_WORDS_SORTED.sort(key=len, reverse=True)

ALL_CORE_WORDS = ('('
                  + '|'.join([re.escape(item) for item in CORE_WORDS_SORTED])
                  + ')')
ALL_KEYWORDS = '(' + '|'.join(KEYWORDS) + ')'


# funcparselib lexer specification
TOKEN_SPECS = [
    ('COMMENT', (r'\( (.|[\r\n])*? \)',)),  # (  ) comments, TODO multiline
    ('COMMENT', (r'\\ .*',)),  # \ comment
    ('NEWLINE', (r'[\r\n]+',)),
    ('SPACE', (r'[ \t\r\n]+',)),
    ('KEYWORD', (ALL_KEYWORDS, re.IGNORECASE)),
    ('CORE_WORD', (ALL_CORE_WORDS, re.IGNORECASE)),
    ('NAME', (r'[A-Za-z_][A-Za-z_0-9\-\+]*',)),
    ('NAME', (r'[0-9]+[A-Za-z_\-\+][\+\-A-Za-z_0-9]*',)),  # starting with nums
    ('REAL', (r'[0-9]+\.[0-9]*([Ee][+\-]?[0-9]+)*',)),
    ('INT', (r'[0-9]+',)),
    ('OP', (r'(\.\.)|(<>)|(<=)|(>=)|(:=)|[;,=\(\):\[\]\.+\-<>\*/@\^]',)),  # TODO, check this out
    # ('STRING', (r"'([^']|(''))*'",)),  # TODO, unnecessary for now
    # ('CHAR', (r'#[0-9]+',)),  # TODO, unnecessary for now
    # ('CHAR', (r'#\$[0-9A-Fa-f]+',)),  # TODO, unnecessary for now
]


def tokenize(code):
    """
    Tokenize the input code with the funcparselib lexer
    (see token_specs for more details)

    For example, code:
        1 1+ DROP
    results in:
        [Token('INT', '1'),
         Token('CORE_WORD', '1+'),
         Token('CORE_WORD', 'DROP'),
         Token('ENDMARKER', '')]

    Args:
        code: code in a string

    Returns:
        list of tokens
    """
    tokenizer = make_tokenizer(TOKEN_SPECS)
    # ignore newlines, spaces and comments
    tokens = [token for token in tokenizer(code)
              if token.type != 'NEWLINE' and
              token.type != 'SPACE' and
              token.type != 'COMMENT']
    end = (tokens[-1].end[0] + 1, 0)
    # add ENDMARKER
    tokens.append(Token('ENDMARKER', '', end, end))
    return tokens


# Constants used to mark the type and value of an AST node.
TYP = 'typ'
VALUE = 'value'


def parse(tokens):
    """
    Parses a sequence of tokens into a Forth AST.

    For example, tokens:
        [Token('INT', '1'),
         Token('CORE_WORD', '1+'),
         Token('CORE_WORD', 'DROP'),
         Token('ENDMARKER', '')]
    result in:
        [{'value': 1, 'typ': 'number'},
         {'value': '1+', 'typ': 'gen_word'},
         {'value': 'drop', 'typ': 'gen_word'}]

    Args:
        tokens: the sequence of `Token` objects that represent the code
                (output of tokenize).

    Returns:
        a Forth AST.

    """
    # Semantic actions and auxiliary functions
    tokval = lambda tok: tok.value
    lower = lambda t: t.lower()

    label = lambda typ: lambda value: {TYP: typ, VALUE: value}

    # test for a literal
    keyword = lambda t: some(lambda tok: tok.value.lower() == t) >> tokval >> label('keyword')

    generic_word = (some(lambda tok: tok.value.lower() not in KEYWORDS and tok.type != 'ENDMARKER')
                    >> tokval >> lower >> label('gen_word'))
    raw_generic_word = (some(lambda tok: tok.value.lower() not in KEYWORDS and tok.type != 'ENDMARKER')
                        >> tokval >> lower)

    def make_number(s):
        """ Helper function for extracting numbers """
        try:
            return int(s)
        except ValueError:
            return float(s)

    def is_int(s):
        """ Helper function to check if token is an integer """
        try:
            int(s)
            return True
        except ValueError:
            return False

    number = (some(lambda tok: tok.type == 'INT' or tok.type == 'REAL')
              >> tokval >> make_number >> label('number'))
    raw_number = (some(lambda tok: tok.type == 'INT' or tok.type == 'REAL')
                  >> tokval >> make_number)
    raw_keyword = lambda t: some(lambda tok: tok.value.lower() == t) >> tokval

    # references to DSTACK, RSTACK and HEAP elements
    dsm_stack_labels = {'D', 'R', 'H'}
    dsm_stack_elem = (some(lambda tok: tok.value[0] in dsm_stack_labels and is_int(tok.value[1:]))
                      >> tokval >> (lambda elem: (elem[0], make_number(elem[1:]))))
    dsm_components = dsm_stack_elem

    # conditionals

    @with_forward_decls
    def ifthen():
        return ((keyword('if') + body + keyword('then'))
                >> (lambda t: {'true_branch': t[1]})
                >> label('ifthen'))

    @with_forward_decls
    def ifthenelse():
        return ((keyword('if') + body + keyword('else') + body + keyword('then'))
                >> (lambda t: {'true_branch': t[1], 'false_branch': t[3]})
                >> label('ifthenelse'))

    # loops

    @with_forward_decls
    def doloop():
        return ((keyword('do') + body + keyword('loop'))
                >> (lambda t: {'body': t[1]})
                >> label('doloop'))

    @with_forward_decls
    def beginwhilerepeat():
        return ((keyword('begin') + body + keyword('while') + body + keyword('repeat'))
                >> (lambda t: {'begin_branch': t[1], 'while_branch': t[3]})
                >> label('beginwhilerepeat'))

    # functions/macros

    @with_forward_decls
    def func():
        return ((keyword(':') + generic_word + body + keyword(';'))
                >> (lambda t: {'body': t[2], 'name': t[1]})
                >> label('def'))

    @with_forward_decls
    def macro():
        return ((keyword('macro:') + generic_word + body + keyword(';'))
                >> (lambda t: {'value': t[2], 'name': t[1]['value']})
                >> label('macro'))

    constant_def = ((number + skip(keyword('constant')) + raw_generic_word)
                    >> (lambda t: {'name': t[1], 'value': [t[0]]})
                    >> label('macro'))

    variable_index_generator = new_label_generator()

    variable_def = ((skip(keyword('variable')) + raw_generic_word)
                    >> (lambda t: {'name': t,
                                   'value': [label('number')(variable_index_generator())]})
                    >> label('macro'))

    create_def = ((skip(keyword('create')) + raw_generic_word + raw_number + skip(keyword('allot')))
                  >> (lambda t: {'name': t[0],
                                 'value': [label('number')(variable_index_generator(t[1]))]})
                  >> label('macro'))

    # slot options

    @with_forward_decls
    def choose():
        return keyword('choose') + many(expr) >> (lambda t: tuple(t[1])) >> label('choose')

    @with_forward_decls
    def manipulate():
        return (keyword('manipulate') + many(dsm_components)
                >> (lambda t: tuple(t[1])) >> label('manipulate'))

    @with_forward_decls
    def permute():
        return (keyword('permute') + many(dsm_components)
                >> (lambda t: tuple(t[1])) >> label('permute'))

    @with_forward_decls
    def observe():
        return (keyword('observe') + many(dsm_components)
                >> (lambda t: tuple(t[1])) >> label('observe'))

    label_generator = new_label_generator('L')

    @with_forward_decls
    def transformed_enc():
        return many(transform + skip(keyword('->'))) >> (lambda t: tuple(t))

    @with_forward_decls
    def preprocessed_dec():
        return many(preprocess + skip(keyword('->'))) >> (lambda t: tuple(t))

    @with_forward_decls
    def slot_only_dec():
        return (keyword('{') + dec + keyword('}')
                >> (lambda t: {'dec': t[1],
                               'enc': {'typ': 'static', 'value': 'None'},
                               'label': label_generator()})
                >> label('slot'))

    @with_forward_decls
    def slot_enc_dec():
        return (skip(keyword('{')) + enc + skip(keyword('->'))
                + transformed_enc + dec + skip(keyword('}'))
                >> (lambda t: {'dec': t[2],
                               'enc': t[0],
                               'label': label_generator(),
                               'transformations': t[1]})
                >> label('slot'))

    @with_forward_decls
    def preprocess():
        raw_keyword('execute') + expr

    # slot transformations
    transform = (raw_keyword('sigmoid') | raw_keyword('tanh')
                 | raw_keyword('linear') + raw_number)
    dec = choose | manipulate | permute  # | sample
    static = keyword('static') >> (lambda t: {'typ': 'static', 'value': 'None'})
    enc = static | observe  # | conjoin  | transformed_enc
    slot = slot_enc_dec | slot_only_dec

    end_marker = a(Token('ENDMARKER', ''))
    newline = a(Token('NEWLINE', '\n'))

    expr = (number | ifthenelse | ifthen | beginwhilerepeat | doloop
            | generic_word | slot | skip(newline))
    body = many(expr)
    prog = (many(macro | constant_def | variable_def | create_def | expr | func)
            + end_marker)

    top_level = prog
    result, _ = top_level.parse(tokens)
    return result


def new_label_generator(prefix=None):
    """
    Label generator - generates an increasing integer-based ID per each call (starts from 0)

    :param prefix: prefix to the generated labels
    :return: a function that generates fresh labels on each call.
    """
    counter = [0]

    def generator(inc_amount=1):
        result = counter[0]
        counter[0] += inc_amount
        return prefix + str(result) if prefix is not None else result

    return generator


non_parallelizable_types = {'doloop', 'beginwhilerepeat'}


def is_collapsable(test_ast):
    """
    Tests whether the AST element can be collapsed.

    :param test_ast: AST to test
    :return: is AST collapsable.
    """
    if isinstance(test_ast, dict):
        if TYP in test_ast and (
                        test_ast[TYP] in non_parallelizable_types or
                        test_ast[TYP] == 'gen_word' and
                        test_ast[VALUE] not in CORE_WORDS):
            return False
        else:
            return all([is_collapsable(value) for value in test_ast.values()])
    elif isinstance(test_ast, list):
        return all([is_collapsable(value) for value in test_ast])
    else:
        return True


def compile_to_im(ast, generate_label=new_label_generator('L'), parallel_branches=False):
    """
    Convert Forth AST into the intermediate representation of a sequence of words.

    For example, ast:
        [{'typ': 'number', 'value': 0}, {'typ': 'gen_word', 'value': '1+'}]
    results in:
        [('CONSTANT', 0), ('1+',)]

    :param ast: AST
    :param generate_label: function that generates fresh labels when called.
    :param parallel_branches: should the compiler generate parallel if branches where possible.
    :return: a sequence of forth words.
    """

    def typ_value_to_tuple(typ_value_dict):
        """ Helper function for converting typ dictionary to (command, value) tuple """
        return typ_value_dict[TYP].upper(), typ_value_dict[VALUE]

    result = []
    for tree in ast:
        if tree[TYP] == 'gen_word':
            if tree[VALUE] in CORE_WORDS:
                result.append(CORE_WORDS[tree[VALUE]])
            else:
                result.append(im.call(tree[VALUE]))

        if tree[TYP] == 'number':
            result.append(im.constant(tree[VALUE]))

        if tree[TYP] == 'ifthen':
            true_branch = tree[VALUE]['true_branch']
            # we can check here whether the true branch has any calls or loops.
            # If not we can create a parallel
            compiled_true_branch = compile_to_im(true_branch, generate_label,
                                                 parallel_branches)
            if parallel_branches and is_collapsable(true_branch):
                result.append(
                    im.parallel([im.drop()] + compiled_true_branch,
                                (im.drop(),)))
            else:
                # IF ... THEN -> branch0(label) ... label
                label = generate_label()
                # print("Label {}".format(label))
                result.append(im.branch0(label))
                result.extend(compile_to_im(true_branch, generate_label, parallel_branches))
                result.append(im.label(label))

        if tree[TYP] == 'ifthenelse':
            true_branch = tree[VALUE]['true_branch']
            false_branch = tree[VALUE]['false_branch']
            compiled_true_branch = compile_to_im(true_branch, generate_label, parallel_branches)
            compiled_false_branch = compile_to_im(false_branch, generate_label, parallel_branches)
            if parallel_branches and is_collapsable(true_branch) and is_collapsable(false_branch):
                result.append(
                    im.parallel([im.drop()] + compiled_true_branch,
                                [im.drop()] + compiled_false_branch))
            else:
                # IF ... ELSE ... THEN -> branch0(l_else) ... branch(l_end) l_else ... lend
                after_then = generate_label()
                after_else = generate_label()
                result.append(im.branch0(after_else))
                result.extend(compiled_true_branch)
                result.append(im.branch(after_then))
                result.append(im.label(after_else))
                result.extend(compiled_false_branch)
                result.append(im.label(after_then))

        if tree[TYP] == 'doloop':
            # DO ... LOOP -> label ... inc(label) terminate
            label = generate_label()
            result.append(im.init_do_loop())
            result.append(im.label(label))
            result.extend(compile_to_im(tree[VALUE]['body'], generate_label, parallel_branches))
            result.append(im.inc_do_loop(label))
            result.append(im.terminate_do_loop())

        if tree[TYP] == 'beginwhilerepeat':
            # BEGIN ... WHILE ... REPEAT -> l_begin ... branch0(l_end) ... branch(l_begin) l_end
            begin_label = generate_label()
            after_repeat = generate_label()
            result.append(im.label(begin_label))
            result.extend(compile_to_im(tree[VALUE]['begin_branch'],
                                        generate_label,
                                        parallel_branches))
            result.append(im.branch0(after_repeat))
            result.extend(compile_to_im(tree[VALUE]['while_branch'],
                                        generate_label,
                                        parallel_branches))
            result.append(im.branch(begin_label))
            result.append(im.label(after_repeat))

        if tree[TYP] == 'def':
            # : xxx .. ; -> branch(l_end) l_xxx ... ; l_end
            label = tree[VALUE]['name'][VALUE]
            after_def = generate_label()
            result.append(im.branch(after_def))
            result.append(im.label(label))
            result.extend(compile_to_im(tree[VALUE]['body'], generate_label, parallel_branches))
            result.append(im.exit())
            result.append(im.label(after_def))

        if tree[TYP] == 'slot':
            dec = tree[VALUE]['dec']
            enc = typ_value_to_tuple(tree[VALUE]['enc'])
            if dec[TYP] == 'choose':
                choices = tuple(compile_to_im(dec[VALUE], generate_label, parallel_branches))
                dec_im = im.choose(choices)
            elif dec[TYP] == 'sample':
                choices = tuple(compile_to_im(dec[VALUE], generate_label, parallel_branches))
                dec_im = im.sample(choices)
            else:
                dec_im = typ_value_to_tuple(dec)
            result.append(im.slot(dec_im, enc, tree[VALUE]['label'],
                                  tree[VALUE].get('transformations', None)))
    return result


def inline_macros(ast):
    """
    Calculates values to macros, constants, and variables, and inline-modifies the rest

    For example, the ast:
        [{'typ': 'macro', 'value': {'value': [{'typ': 'number', 'value': 0}], 'name': 'question'}},
         {'typ': 'gen_word', 'value': 'question'}, {'typ': 'gen_word', 'value': '1+'}]
    results in:
        [{'typ': 'number', 'value': 0}, {'typ': 'gen_word', 'value': '1+'}]

    of the code to incorporate these values.
    :param ast: AST
    :return: inline-modified AST
    """

    # collect macros (constants variables etc) and remove them
    macros = {}

    def collect(element):

        # if element refers to a macro, expand and create a list
        if isinstance(element, dict) and TYP in element and element[TYP] == 'macro':
            binding = element[VALUE]['value']
            macros[element[VALUE]['name']] = collect(binding)
            return []

        # if elem is a gen_word and in collection of macros, return its value
        elif (isinstance(element, dict)
              and TYP in element
              and element[TYP] == 'gen_word'
              and element[VALUE] in macros):
            return macros[element[VALUE]]

        # otherwise, if elem is a collection, go over each elem
        elif isinstance(element, list):
            result = []
            for e in element:
                inlined = collect(e)
                if isinstance(inlined, list):
                    result += inlined
                else:
                    result.append(inlined)
            return result

        elif isinstance(element, dict):
            result = {key: collect(value) for key, value in element.items()}
            return result

        else:
            return element

    return collect(ast)


def collapse_im(im_code):
    """
    Collapses sub-sequences of forth words. Splits at branches, calls and labels.

    For example, the code:
        (('BRANCH', 1),
         ('LABEL', 0),
         ('1+',),
         ('1+',),
         ('EXIT',),
         ('LABEL', 1),
         ('>R',),
         ('CALL', 0))
    results in:
        [('MACRO', (('BRANCH', 1),)),
         ('LABEL', 0),
         ('MACRO', (('1+',), ('1+',), ('EXIT',))),
         ('LABEL', 1),
         ('MACRO', (('>R',), ('CALL', 0)))]

    :param im_code: the original sequence of forth words.
    :return: sequence of words where some words are themselves sequences of core words.
    """
    result = []
    current = []

    def append_current():
        if current:
            result.append(im.macro(tuple(current)))
            current.clear()

    for index, word in enumerate(im_code):
        if word[0] == im.label()[0]:
            append_current()
            result.append(word)

        elif word[0].startswith("BRANCH"):
            current.append(word)
            append_current()

        elif word[0] == im.parallel()[0]:
            append_current()
            left, right = word[1:]
            result.append(im.parallel(collapse_im(left), collapse_im(right)))

        elif word[0] == im.macro()[0]:
            append_current()
            result.append(im.macro(collapse_im(word[1])))

        elif word[0] == im.call()[0]:
            current.append(word)
            append_current()

        elif word[0] == im.inc_do_loop()[0]:
            current.append(word)
            append_current()

        elif word[0] == im.slot()[0]:
            append_current()
            result.append(word)
            # current.append(word)
            # append_current()
        else:
            current.append(word)

    append_current()
    return result


def compile(source, parallel_branches=False):
    """
    Compile the source code from a string into intermediate code

    For example, source:
        1 2 { observe D-1 D0 -> manipulate D-1 } DROP
    results in:
        [('CONSTANT', 1),
         ('CONSTANT', 2),
         ('SLOT', ('MANIPULATE', (('D', -1),)), ('OBSERVE', (('D', -1), ('D', 0))), 'L0', ()),
         ('DROP',)]

    Args:
        source: textual source code
        parallel_branches: generate parallel if branches

    Returns:
        intermediate code
    """
    logging.debug("Source: {}".format(source))

    # tokenize
    tokens = tokenize(source)
    logging.debug("Tokens: {}".format(tokens))

    # parse into ast
    ast = parse(tokens)
    logging.info("AST: {}".format(ast))

    # calculates constants & stuff and inlines them
    inlined = inline_macros(ast)
    logging.info("Inlined: {}".format(inlined))

    # compile to intermediate code
    im_code = compile_to_im(inlined, parallel_branches=parallel_branches)
    logging.info("IM: {}".format(im_code))

    return im_code


def main():
    logging.basicConfig(level=logging.DEBUG)
    code = "1 2 { observe D-1 D0 -> manipulate D-1 } DROP"
    print(compile(code))


if __name__ == "__main__":
    main()
