import os
import sys
import pandas as pd
import torch
from metrics import Critic

if __name__ == '__main__':

    _ = open(sys.argv[1], 'w')

    def write(s):

        with open(sys.argv[1], 'a') as csv:
            csv.write(s + '\n')
    
    max_folds = 5
    max_trials = 5
    configs = sys.argv[2:]

    # NOTE: In python 3.7 an onward insertion order 
    # is a language specification. This is the order of 
    # insertion in metrics.py and subsequently evaluate.py
    # get insertion order used in evaluate
    critic = Critic()
    m = critic.list_template()
    m = {f'{mode}-{k}' : None
        for k in m.keys() 
        for mode in ['FL', 'T1']}
    m['FL-AUC'] = []; m['T1-AUC'] = []
    m['FL-PRC'] = []; m['T1-PRC'] = []
    mkeys = ','.join(m.keys())

    write(f'model,epoch,fold,trial,{mkeys}')
    for config in configs:
        for fold in range(max_folds):
            for trial in range(max_trials):
                p = f'{config}/fold-{fold}/trial-{trial}/evaluations/results.txt'
                try:
                    with open(p, 'r') as out:
                        line = list(out.readlines())[0].strip()
                        model_name = config
                        # epoch = -1
                        epoch = torch.load(f'{config}/fold-{fold}/trial-{trial}/model.pt')['epoch']
                        s = f'{model_name},{epoch},{fold},{trial}'
                        for part in line.split('='):
                            try:
                                num = float(part.split('|')[0])
                            except:
                                continue
                            s += f',{num}'
                        write(s)
                except Exception as e:
                    continue
    df = pd.read_csv(sys.argv[1])
