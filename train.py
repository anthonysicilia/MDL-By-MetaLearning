import datasets
import procedures
import optimizers
import losses
import metrics
import models
import torch
import sys

from evaluate import run as post_eval
from runner import Runner

if __name__ == '__main__':

    NUM_TRIALS = 5

    config_loc = sys.argv[1]

    if len(sys.argv) > 2:
        gpu = int(sys.argv[2])
    else:
        gpu = 0
    
    if len(sys.argv) > 3:
        NUM_TRIALS = int(sys.argv[3])

    runner = Runner(models=models.MODELS,
        dsets=datasets.DATASETS,
        losses=losses.LOSSES,
        optims=optimizers.OPTIMIZERS,
        procedures=procedures.PROCEDURES,
        critic=metrics.Critic(),
        post_eval=post_eval)
    
    for t in range(NUM_TRIALS):
        runner.run(config_loc, trial=t, gpu=gpu)
    
