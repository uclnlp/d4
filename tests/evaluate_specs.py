import tensorflow as tf

import d4.dsm.extensible_dsm as edsm
import d4.interpreter as si
from d4.dsm.loss import CrossEntropyLoss

eps = 0.00001


def evaluate_specs(stack_size, value_size, min_return_width,
                   steps, specs, debug=False, parallel_branches=True):
    interpreter = si.SimpleInterpreter(stack_size, value_size, min_return_width,
                                       len(specs), parallel_branches=parallel_branches)

    for batch, (_, _, code) in enumerate(specs):
        interpreter.load_code(code, batch)

    trace = interpreter.execute(steps)
    loss = CrossEntropyLoss(trace[-1], interpreter)

    for batch, (input_, output_, _) in enumerate(specs):
        interpreter.load_stack(input_, batch)
        loss.load_target_stack(output_, batch)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    result = sess.run(loss.loss, loss.current_feed_dict())

    if debug:
        for i in range(0, steps):
            print("-" * 50)
            print("Step {}".format(i))
            print("Data Stack")
            state = trace[i]
            print(sess.run(edsm.pretty_print_buffer(state.data_stack),
                           interpreter.current_feed_dict()))
            print("Data Stack Pointer")
            print(sess.run(edsm.pretty_print_value(state.data_stack_pointer),
                           interpreter.current_feed_dict()))
            print("Return Stack")
            print(sess.run(edsm.pretty_print_buffer(state.return_stack),
                           interpreter.current_feed_dict()))
            print("Return Stack Pointer")
            print(sess.run(edsm.pretty_print_value(state.return_stack_pointer),
                           interpreter.current_feed_dict()))
            # print("Heap")
            # print(sess.run(edsm.pretty_print_buffer(state.heap), interpreter.current_feed_dict()))
            print("PC")
            pc = sess.run(edsm.pretty_print_value(state.pc), interpreter.current_feed_dict())
            print(pc)
            # print("Current Word:")
            # for batch in range(0, interpreter.batch_size):
            #     print("Batch {}".format(batch))
            #     for word_index in range(0, state.code_size):
            #         score = pc[batch, word_index]
            #         if score > 0.5:
            #             word = interpreter.words[interpreter.final_code[batch][word_index]]
            #             print("{} {}".format(score, word))

        print("-" * 50)
        print("Loss: {}".format(result))
        print("Target")
        print(sess.run(edsm.pretty_print_buffer(loss.data_stack_target_placeholder),
                       loss.current_feed_dict()))
        print(sess.run(edsm.pretty_print_value(loss.data_stack_pointer_target_placeholder),
                       loss.current_feed_dict()))

    assert abs(result) < eps
