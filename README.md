# ∂4, the Differentiable Forth Interpreter

This is the implementation of ∂4, as presented in the [ICML 2017 paper](http://proceedings.mlr.press/v70/bosnjak17a.html).


BibTex entry:
```
@inproceedings{bosnjak17,
    title =     {Programming with a Differentiable Forth Interpreter},
    author =    {Matko Bo{\v{s}}njak and Tim Rockt{\"a}schel and Jason Naradowsky and Sebastian Riedel},
    booktitle = {Proceedings of the 34th International Conference on Machine Learning},
    year =      {2017}
}
```


## Disclaimer

Please be aware that this is highly experimental research code, thus not well commented, and possibly difficult to follow. The code is not being actively maintained and we provide no warranty for its use.


## Installation

### Dependencies

Python 3

with

- Tensorflow 0.11.0

```
pip3 install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp35-cp35m-linux_x86_64.whl
```

- funcparserlib

```
pip3 install funcparserlib
```

### Then

...then just git pull the repo.


## Use

### Datasets (and generation)

All the datasets are already included in the repo and there is no need to recreate them.
The datasets can be found in the data/ directory.

However, should you want to recreate them, use the following scripts:

#### Adder

    python3 data/add_generate_data.py

#### Bubble sort

    python3 data/sort_generate_data.py

#### Word Algebra Problems

This one is a bit trickier as it requires CoreNLP tokenizer.

First install grequests:

    pip3 install grequests

Then download CoreNLP:

    source util/corenlp/download_corenlp.sh

Then run it:

    source util/corenlp/run_corenlp.sh

While the CoreNLP service is running, generate the data:

    python3 data/wap_generate_data.py


### Running the experiments

#### Bubble sort

```
python3 experiments/sort/sort.py \
    --sketch experiments/sort/sketch_compare.d4 \
    --data data/sort/train_test_len/train2_test64/ \
    --learning_rate 1.0 \
    --batch_size 32 \
    --max_epochs 500 \
    --init_weight_stddev 0.1 \
    --max_grad_norm 1.0
```

Available sketches:

* experiments/sort/sketch_compare.d4
* experiments/sort/sketch_permute.d4

#### Adder

```
python3 experiments/add/add.py \
    --sketch experiments/add/sketch_manipulate.d4 \
    --data data/add/train_test_len/train8_test64/ \
    --learning_rate 0.05 \
    --batch_size 32
```

Available sketches:
* experiments/add/sketch_choose.d4
* experiments/add/sketch_manipulate.d4

#### Word Algebra Problems

```
python3 experiments/wap/trainer.py
```


## ∂4 Operators

Please find the description of the implemented commands/operators [here](http://proceedings.mlr.press/v70/bosnjak17a/bosnjak17a-supp.pdf) (in the paper's appendix).


## Other useful stuff

[One list of Forth operations](http://astro.pas.rochester.edu/Forth/forth-words.html) and [another one](http://www.wulfden.org/downloads/Forth_Resources/SP_ProgrammingForth.pdf#page=29)

[Forth REPL](https://repl.it/languages/forth)


## Tips
1. **INCREASE THE STEP SIZE** !
2. **INCREASE THE STACK SIZE** !!!
3. Increase the minimum return width
4. Forth REPL needs RECURSE keyword instead of a recursive function call
5. When encountering bugs, keep in mind there are now two sources of error. TensorFlow code and Forth scaffolds/code :)
