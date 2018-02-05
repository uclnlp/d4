import numpy as np
import matplotlib
# matplotlib.use('Svg')

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

from matplotlib.backends.backend_pdf import PdfPages

from d4.dsm.extensible_dsm import (PipelineWord, EncoderDecoderWord)


def visualise_pc_trace(pcs, code=None, filename=None, filename_suffix=''):
    # pcs = np.ones(np.shape(pcs)) - pcs
    # code_rev = code[::-1]
    # pcs_flip = np.fliplr(pcs)

    fig = plt.figure(figsize=(20, 20), dpi=500)
    fig.patch.set_alpha(0.0)

    # fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111)
    # ax.patch.set_alpha(0.0)

    ax.matshow(np.transpose(pcs), cmap=plt.cm.gray)

    # ax.grid(False)
    # ax.axis('off')
    ax.set_yticks(range(len(pcs[0])))
    ax.tick_params(axis='both', which='both', length=0)
    # ax.xaxis.set_visible(False)
    ax.set_xticks(range(len(pcs)))

    if code is not None:
        ax.set_yticklabels(code)
    else:
        ax.set_yticklabels([''] * 20)

    ax.xaxis.label.set_color('red')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    import matplotlib.ticker as plticker
    loc = plticker.MultipleLocator(base=10.0)
    ax.xaxis.set_major_locator(loc)

    if filename is not None:
        # pp = PdfPages(filename + filename_suffix + '.svg')
        pp = filename + filename_suffix + '.png'
        plt.savefig(pp, format='png', bbox_inches='tight')
        # pp.close()
    else:
        plt.show()


def extract_commands(interpreter):
    code = []
    for item in interpreter.vocab.words:
        if isinstance(item, PipelineWord) or isinstance(item, EncoderDecoderWord):
            code.append('*')
            continue
        # if item.word[0] != 'MACRO':
        #     print(item.word[0])
        if item.word[0] != 'HALT':
            code.append(" ".join([it[0] for it in item.word[1]]))
        else:
            code.append("HALT")
    return code


def plot_heap(stack, pointer=[], flip=True):
    mtrx = stack
    pntr = np.transpose([pointer])
    if flip:
        mtrx = np.flipud(stack)
        pntr = np.flipud(np.transpose([pointer]))
    fig = plt.figure()
    if len(pntr > 0):
        gs = gridspec.GridSpec(1, 2)
    else:
        gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(mtrx, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)

    if len(pntr > 0):
        ax2 = fig.add_subplot(gs[0, 1])

    if len(pntr > 0):
        im = ax2.imshow(pntr, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)

    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    ax1.set_yticklabels(reversed(range(np.size(stack, axis=0))))
    ax1.set_xticks(range(np.size(stack, axis=0)))
    ax1.set_yticks(range(np.size(stack, axis=1)))

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    plt.colorbar(im)
    plt.show()


# def prettyprint_vector_code(vector_code: VectorCode):
#     decoded = vector_code.target_language.decode_vector_code(vector_code)
#
#     vocab_inv = {v: k for k, v in vector_code.vocab.items()}
#     for i, line in enumerate(vector_code.current_source):
#         cmd = decoded[i]
#         name = vocab_inv.get(i, '')
#         start = ''
#         if i == vector_code.entry:
#             start = "entr ->"
#         print("{0}\t{1}\t{2}".format(start, name, cmd))
# #         print(interpreter.vector_code.annotations)
#
#

# def visualize_trace_diff(full_trace):
#     trace_length = len(full_trace[0])
#     code = full_trace[6]
# #     print(code)
#     var_dict = {
#         0: "PC",
#         1: "DSTACK",
#         2: "DSTACK PTR",
#         3: "RSTACK",
#         4: "RSTACK PTR"
#     }
#
#
#     print("[ Iteration 0 ]",'\n','--'*50)
#     for i in range(5):
#         print(var_dict[i],'\n', full_trace[i][0])
#
#
#     for i in range(1, trace_length):
#         print('--'*50, '\n', "[ Iteration {} ]".format(i), '\n', '--'*50)
#         for j in range(5):
#             if not (full_trace[j][i-1] == full_trace[j][i]).all():
#                 print(var_dict[j], 'CHANGE:','\n',full_trace[j][i])
