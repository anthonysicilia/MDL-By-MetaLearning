import argparse
from pathlib import Path
import json
import os

# times 2 for cyclic indexing 
FL_PATHS = 2 * [
    "paths/fold-0/FL_paths_ws.txt",
    "paths/fold-1/FL_paths_ws.txt",
    "paths/fold-2/FL_paths_ws.txt",
    "paths/fold-3/FL_paths_ws.txt",
    "paths/fold-4/FL_paths_ws.txt",
]

T1_PATHS = 2 * [
    "paths/fold-0/T1_paths_ws.txt",
    "paths/fold-1/T1_paths_ws.txt",
    "paths/fold-2/T1_paths_ws.txt",
    "paths/fold-3/T1_paths_ws.txt",
    "paths/fold-4/T1_paths_ws.txt",
]

LABEL_PATHS = 2 * [
    "paths/fold-0/label_paths.txt",
    "paths/fold-1/label_paths.txt",
    "paths/fold-2/label_paths.txt",
    "paths/fold-3/label_paths.txt",
    "paths/fold-4/label_paths.txt",
]

def variants(agg):
    v = [
        {
            'name' : 'baseline',
            'update_procedure' : {
                'name' : 'standard_update',
                'params' : {
                    'agg' : agg,
                    'proportional' : False,
                    'lamb' : None
                }
            },
        },
        {
            'name' : 'baseline-10',
            'update_procedure' : {
                'name' : 'standard_update',
                'params' : {
                    'agg' : agg,
                    'proportional' : False,
                    'lamb' : 0.10
                }
            },
        },
        {
            'name' : 'baseline-90',
            'update_procedure' : {
                'name' : 'standard_update',
                'params' : {
                    'agg' : agg,
                    'proportional' : False,
                    'lamb' : 0.90
                }
            },
        }
    ]

    for w in [10, 25, 100]:
        v.append({
            'name' : f'map-GT-{w}',
            'update_procedure' : {
                'name' : 'moving_update',
                'params' : {
                    'agg' : agg,
                    'style' : 'map',
                    'hypers' : {
                        'direction' : '>',
                        'window_size' : w
                    }
                }
            },
        })
    
    v.append({
            'name' : f'simple-GT-{w}',
            'update_procedure' : {
                'name' : 'moving_update',
                'params' : {
                    'agg' : agg,
                    'style' : 'simple',
                    'hypers' : {
                        'direction' : '>'
                    }
                }
            },
    })

    for w in [10, 25, 100]:
        v.append({
            'name' : f'map-LT-{w}',
            'update_procedure' : {
                'name' : 'moving_update',
                'params' : {
                    'agg' : agg,
                    'style' : 'map',
                    'hypers' : {
                        'direction' : '<',
                        'window_size' : w
                    }
                }
            },
        })
    
    v.append({
            'name' : f'simple-LT-{w}',
            'update_procedure' : {
                'name' : 'moving_update',
                'params' : {
                    'agg' : agg,
                    'style' : 'simple',
                    'hypers' : {
                        'direction' : '<'
                    }
                }
            },
    })
    
    return v

def generate(model, num_fl, agg, lr, max_gpu, num_trials):
    agg = bool(agg)
    agg_d = 'R' if agg else 'DT'
    lr_d = f'-{lr}' if lr != 0.01 else ''
    save_dir = f'{model}-{agg_d}-{num_fl}fl{lr_d}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for v in variants(agg):

        template = dict()

        template['model'] = {
            'name' : model,
            'params' : {}
        }

        template['loss'] = {
            'name' : 'dsc-loss',
            'params' : {
                'dsc_loss_coeff' : 1.0
            }
        }

        template['optim'] = {
            'name' : 'sgd',
            'params' : {
                'lr' : lr,
                'momentum' : 0.9
            }
        }

        template['update_procedure'] = v['update_procedure']
        if template['update_procedure']['name'] == 'moving_update':
            template['uproc_state_init'] = 'moving_init'

        template['validation_procedure'] = {
            'name' : 'validate',
            'params' : {}
        }

        template['max_epochs'] = 500
        template['patience'] = 20
        template['parallel'] = False
        template['folds'] = [
            {
                'trainset' : {
                    'name' : 'mixed',
                    'params' : {
                        'fl_paths' : FL_PATHS[i+1:i+4],
                        't1_paths' : T1_PATHS[i+1:i+4],
                        'label_paths' : LABEL_PATHS[i+1:i+4],
                        'augment' : True,
                        'num_fl' : num_fl,
                        'agg' : agg
                    },
                    'loader_params' : {
                        'batch_size' : 8 if agg else 4,
                        'drop_last' : True,
                        'num_workers' : 4,
                        'shuffle' : True
                    }
                },
                'valset' : {
                    'name' : 'paired',
                    'params' : {
                        'fl_paths' : [FL_PATHS[i+4]],
                        't1_paths' : [T1_PATHS[i+4]],
                        'label_paths' : [LABEL_PATHS[i+4]],
                        'augment' : False
                    },
                    'loader_params' : {
                        'batch_size' : 5,
                        'drop_last' : False,
                        'num_workers' : 4,
                        'shuffle' : False
                    }
                },
                'testset' : {
                    'name' : 'paired',
                    'params' : {
                        'fl_paths' : [FL_PATHS[i+5]],
                        't1_paths' : [T1_PATHS[i+5]],
                        'label_paths' : [LABEL_PATHS[i+5]],
                        'augment' : False
                    },
                    'loader_params' : {
                        'batch_size' : 5,
                        'drop_last' : False,
                        'num_workers' : 4,
                        'shuffle' : False
                    }
                }
            } for i in range(5)
        ]

        _save_dir = f'{save_dir}/{v["name"]}'
        Path(_save_dir).mkdir(parents=True, exist_ok=True)
        with open(f'{_save_dir}/config.json', 'wt') as out:
            json.dump(template, out, 
                indent=4, 
                separators=(',', ': '))
        
    with open(f'{save_dir}/run.sh', 'w') as out:

        gpu = 0
        for v in variants(agg):
            name = f'{save_dir}/{v["name"]}'
            inner = f'train.py {name} {gpu} {num_trials}'
            out.write(f'python3 {inner} > catch.txt &\n')
            out.write(f'PID{gpu}=$!\n')
            gpu += 1
            if gpu >= max_gpu:
                gpu = 0
                out.write('wait ' + ' '.join(
                    [f'$PID{g}' for g in range(max_gpu)])
                    + '\n')
        out.write('wait ' + ' '.join(
            [f'$PID{g}' for g in range(gpu)])
            + '\n')
        out.write(f'python3 write_results.py {save_dir}/results.txt '
            + ' '.join([f'{save_dir}/{v["name"]}' 
            for v in variants(agg)])
            + '\n')
        out.write(f'python3 summarize.py {save_dir}/results.txt')
        
    os.chmod(f'{save_dir}/run.sh', 777)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Model to use',
        type=str, default='unet')
    parser.add_argument('-fl', '--num_fl', help='Number of Flair',
        type=int, default=12)
    parser.add_argument('-lr', '--lr', help='Learning Rate to use',
        type=float, default=0.01)
    parser.add_argument('-r', '--agg', 
        help='Within batch (modality) randomness',
        type=int, default=1)
    parser.add_argument('-g', '--max_gpu', help='Number of gpus',
        type=int, default=4)
    parser.add_argument('-t', '--num_trials', help='Number of trials',
        type=int, default=5)
    args = parser.parse_args()

    generate(**vars(args))
