import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from pathlib import Path
from torchvision.transforms.functional import to_pil_image, \
    vflip, to_tensor
from sklearn.metrics import roc_auc_score, average_precision_score
from datasets import Paired
from metrics import Critic
from models import MODELS

def run(model_path, gpu=0):

    parent_dir = model_path.split('/model')[0]
    config_loc = parent_dir.split('/fold')[0]

    with open(f'{parent_dir}/log.txt', 'r') as log:

        val_curve = []
        train_curve = []

        for b, line in enumerate(log.readlines()):

            if 'validation' in line:

                storage = val_curve
            else:

                storage = train_curve
            
            loss = float(line.split(f'loss=')[-1].split('|')[0])

            storage.append((b, loss))

        plt.clf()
        plt.plot([a for a,b in train_curve], 
            [b for a,b in train_curve], label='train')
        plt.plot([a for a,b in val_curve], 
            [b for a,b in val_curve], label='validation')
        plt.xlabel('batches')
        plt.ylabel('loss')
        plt.title(f'learning curves')
        plt.legend()
        plt.savefig(f'{parent_dir}/learning_curves')


    with open(f'{config_loc}/config.json', 'r') as config:
        try:
            params = json.load(config)
        except Exception as e:
            exit(e)
    
    save_dir = f'{parent_dir}/evaluations'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu}" if use_cuda else "cpu")

    checkpoint = torch.load(model_path, 
        map_location=device)

    model_name = params['model']['name']
    model_params = params['model']['params']
    model = MODELS[model_name](**model_params)

    model.load_state_dict(checkpoint['model_states'])
    
    parallel = True
    if 'parallel' in params:
        parallel = params['parallel']
    
    model = model.to(device)
    if parallel:
        model = torch.nn.DataParallel(model)
    
    fold = int(model_path.split('fold-')[-1].split('/')[0])
    
    fold_params = params['folds'][fold]
    
    # some fixed assumptions on test set for evaluation
    testset_params = fold_params['testset']['params']
    testloader_params = {
        "batch_size" : 5,
        "drop_last" : False,
        "num_workers" : 0,
        "shuffle" : False
    }

    test = torch.utils.data.DataLoader(
        Paired(**testset_params),
        **testloader_params
    )

    modes = ['FL', 'T1']
    critic = Critic()
    running_metrics = critic.list_template()
    running_metrics = {f'{mode}-{k}' : [] 
        for k in running_metrics.keys() 
        for mode in modes}
    running_metrics['FL-AUC'] = []
    running_metrics['T1-AUC'] = []
    running_metrics['FL-PRC'] = []
    running_metrics['T1-PRC'] = []

    with torch.no_grad():

        model.eval()

        for subject in test:

            for mode in modes:

                x = subject[mode].to(device)
                y = subject['label'].to(device)
                scores = model(x)
                yhat = (scores.sigmoid() > .5).long()
                metrics = critic(scores, y)
                auc = roc_auc_score(y.view(-1).cpu(),
                    scores.sigmoid().view(-1).cpu())
                prc = average_precision_score(y.view(-1).cpu(),
                    scores.sigmoid().view(-1).cpu())
                running_metrics[f'{mode}-AUC'].append(auc)
                running_metrics[f'{mode}-PRC'].append(prc)
                for k,v in metrics.items():
                    running_metrics[f'{mode}-{k}'].append(v)
                # testloader should return one subject at a time
                assert len(set(subject['subject'])) == 1
                subj_name = subject['subject'][0]
                path = f'{save_dir}/{subj_name}-{mode}.txt'
                with open(path, 'w') as out:
                    out.write('|'.join([f'{k}={v:.4f}' for
                        k,v in metrics.items()]) + '\n')

                for j in range(y.size(0)):
                    
                    # NOTE: important to convert to numpy float array
                    # to_pil_image makes assumptions based on input when mode = None
                    # e.g. see tools.augment for more details
                    
                    fig, ax = plt.subplots(1, 3, figsize=(10,6))

                    predicted = yhat[j, 0].transpose(0,1
                        ).cpu().numpy().astype('float32')
                    actual = y[j, 0].transpose(0,1
                        ).cpu().numpy().astype('float32')
                    inpt = x[j, 0].transpose(0,1
                        ).cpu().numpy().astype('float32')
                    predicted = to_tensor(vflip(to_pil_image(
                        predicted, mode='F'))).float().squeeze(0)
                    actual = to_tensor(vflip(to_pil_image(actual, 
                        mode='F'))).float().squeeze(0)
                    inpt = to_tensor(vflip(to_pil_image(inpt, 
                        mode='F'))).float().squeeze(0)
                    ax.flat[0].imshow(inpt, cmap='gray')
                    ax.flat[0].set_title(f'{mode}')
                    ax.flat[1].imshow(predicted)
                    ax.flat[1].set_title('Predicted')
                    ax.flat[2].imshow(actual)
                    ax.flat[2].set_title('Actual')
                    plt.tight_layout()
                    plt.savefig(f'{save_dir}/{subj_name}-{mode}-{j}')
                    plt.close(fig)

                    fig, ax = plt.subplots()
                    plt.imshow(inpt, cmap='gray', label=f'{mode}')
                    masked_actual = np.ma.masked_where(
                        actual < 0.9, actual)
                    masked_predicted = np.ma.masked_where(
                        predicted < 0.9, predicted)
                    plt.imshow(masked_actual, cmap='Reds', 
                        label='actual', vmin=0)
                    plt.imshow(masked_predicted, cmap='Blues', 
                        label='predicted', vmin=0, alpha=.7)
                    ax.text(-5, 5, 
                        '(blue = prediction) \n (red = actual)', 
                        bbox={'facecolor': 'white', 'pad': 10})
                    plt.savefig(f'{save_dir}/{subj_name}-{j}-{mode}'
                        '-overlay')
                    plt.close(fig)

        for k,v in running_metrics.items():
            running_metrics[k] = sum(v) / len(test)

        with open(f'{save_dir}/results.txt', 'w') as out:
            out.write('|'.join([f'{k}={v:.4f}' for
                k,v in running_metrics.items()]) + '\n')

if __name__ == '__main__':

    run(sys.argv[1])

