# MDL-By-MetaLearning
This is the repository for the paper "Multi-Domain Learning by Meta-Learning: Taking Optimal Steps in Multi-Domain Loss Landscapes by Inner-Loop Learning"
published in ISBI 2021. Available at: https://arxiv.org/abs/2102.13147

Note, the data used in the paper cannot be made public.
In any case, if you would like to run the code on a different dataset,
we provide some details on how this might be done.

`setup_experiments.py` may be used to generate config files
required to run experiments. File paths declared at the
start should be followed or altered appropriately.
We provide a (made-up) example to illustrate 
the appropriate contents of these file paths in `fake-paths/example-fold`.

Following this, the `run.sh` bash script contained in the 
generated experiment folder can be run to do all
experiments listed in the folder (each experiment 
will be associated with its own config directory).
We provide an example of the default experiment folder output
by this script in `example-experiment-config`.
Depending on the number of available GPUs in your system,
you may need to modify the <GPU_IDX>
scheme assumed in this bash script
(more details on how to modify calls to `train.py` below).

If you want to run experiments using your 
own config files, you can do so by 
calling the following in your terminal:

`python3 train.py <CONFIG_DIRECTORY> <GPU_IDX> <NUM_TRIALS>`

You can follow the appropriate calls if your 
research requires modifying our pipeline.
Most of the logic is contained in 
`trainer.py` and `runner.py` with other 
python files containing the necessary 
helper functions.

If you have any questions, don't hesitate to 
post an issue here or reach out at our emails
(available in the paper). We will 
not actively maintain the codebase,
but are happy to help in this capacity.

Finally, if you make use of our code
or techniques, please cite our paper.

Thanks.



