import json
import torch
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Runner:

    def __init__(self, models, dsets, losses, optims, procedures, 
        critic, post_eval=None):
        
        self.models = models
        self.datasets = dsets
        self.losses = losses
        self.optimizers = optims
        self.procedures = procedures
        self.critic = critic
        self.post_eval = post_eval
    
    def run(self, config_loc, trial=0, gpu=0):

        config_f = f'{config_loc}/config.json'

        with open(config_f, 'r') as config:
            try:
                params = json.load(config)
            except Exception as e:
                exit(e)
        
        use_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{gpu}" if use_cuda else "cpu")

        model_name = params['model']['name']
        model_params = params['model']['params']
        loss_name = params['loss']['name']
        loss_params = params['loss']['params']
        optim_name = params['optim']['name']
        optim_params = params['optim']['params']

        uproc_params = params['update_procedure']
        update_procedure = self.procedures[uproc_params['name']]
        uproc_params = uproc_params['params']
        
        vproc_params = params['validation_procedure']
        validation_procedure = self.procedures[vproc_params['name']]
        vproc_params = vproc_params['params']
        
        parallel = True
        if 'parallel' in params:
            parallel = params['parallel']
        
        Model = self.models[model_name]
        Loss = self.losses[loss_name]
        Optim = self.optimizers[optim_name]

        loss_fn = Loss(**loss_params)
        
        max_epochs = params['max_epochs']
        patience = params['patience']

        if 'global_warmstart' in params:
            global_warmstart = params['global_warmstart']
        else:
            global_warmstart = None

        for K, fold_params in enumerate(params['folds']):

            fold_dir = f'fold-{K}/trial-{trial}'

            Path(f'{config_loc}/{fold_dir}').mkdir(parents=True, 
                exist_ok=True)
            log_f = f'{config_loc}/{fold_dir}/log.txt'
            save_dir = f'{config_loc}/{fold_dir}'
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            model = Model(**model_params)

            if 'uproc_state_init' in params:

                uproc_init = params['uproc_state_init']

                if uproc_init is not None:

                    self.procedures[uproc_init](uproc_params)

            optim = Optim(model.parameters(), **optim_params)

            scheduler = ReduceLROnPlateau(optim, patience=patience)

            warmstart = None

            if global_warmstart is not None:

                if 'SMART_MATCH' in global_warmstart:
                    # infer model path based on parent directory
                    dir_path = global_warmstart.split(':')[-1]
                    warmstart = f'{dir_path}/{fold_dir}/model.pt'
                else:
                    raise NotImplementedError('Only SMART_MATCH' \
                        ' global warmstart is implemented')
            
            if 'warmstart' in fold_params:
                # overwrites global warm start if exists
                warmstart = fold_params['warmstart']
            
            if warmstart is not None:
                warmstart = torch.load(warmstart, 
                    map_location=device)
                model.load_state_dict(warmstart['model_states'])
                optim.load_state_dict(warmstart['optim_states'])
                for state in optim.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)

            model = model.to(device)
            if use_cuda and parallel:
                model = torch.nn.DataParallel(model)

            trainset_name = fold_params['trainset']['name']
            trainset_params = fold_params['trainset']['params']
            trainloader_params = \
                fold_params['trainset']['loader_params']
            valset_name = fold_params['valset']['name']
            valset_params = fold_params['valset']['params']
            valloader_params = \
                fold_params['valset']['loader_params']
            testset_name = fold_params['testset']['name']
            testset_params = fold_params['testset']['params']
            testloader_params = \
                fold_params['valset']['loader_params']

            Trainset = self.datasets[trainset_name]
            Valset = self.datasets[valset_name]
            Testset = self.datasets[testset_name]

            train = torch.utils.data.DataLoader(
                Trainset(**trainset_params), 
                **trainloader_params
            )

            val = torch.utils.data.DataLoader(
                Valset(**valset_params), 
                **valloader_params
            )

            test = torch.utils.data.DataLoader(
                Testset(**testset_params), 
                **testloader_params
            )

            best_loss = None
            best_epoch = None

            for epoch in range(0, max_epochs):

                model.train()

                for i, batch in enumerate(train):

                    for k,v in batch.items():
                        if torch.is_tensor(v):
                            batch[k] = v.to(device)

                    metrics = update_procedure(model, batch, 
                        optim, loss_fn, **uproc_params)
                    
                        
                    s = f'{100 * i / len(train) : .0f}%: '

                    with open(log_f, 'a') as log:

                        log.write(s + '|'.join([f'{k}={v:.4f}' 
                            if type(v) is not str else f'{k}={v}' 
                            for k,v in metrics.items()]) + '\n')

                with torch.no_grad():

                    model.eval()
                    metrics = self.critic.list_template()
                    metrics['loss'] = []

                    for batch in val:

                        for k,v in batch.items():
                            if torch.is_tensor(v):
                                batch[k] = v.to(device)

                        _metrics = validation_procedure(model, 
                            batch, loss_fn, self.critic, 
                            **vproc_params)

                        try:

                            for k,v in _metrics.items():
                                metrics[k].append(v)

                        except KeyError:

                            # procuedre metrics don't match
                            # overwrite the template

                            metrics = {k : [] for k in 
                                _metrics.keys()}
                            
                            for k,v in _metrics.items():
                                metrics[k].append(v)
                    
                for k,v in metrics.items():

                    metrics[k] = sum(metrics[k]) / len(val)
                
                with open(log_f, 'a') as log:

                    log.write(f'validation {epoch}: ' 
                        + '|'.join([f'{k}={v:.4f}' for 
                            k,v in metrics.items()]) + '\n')
                
                val_loss = metrics['loss']
                scheduler.step(val_loss)

                if best_loss is None or val_loss < best_loss:
                    best_epoch = epoch
                    best_loss = val_loss
                    if use_cuda and parallel:
                        msd = model.module.state_dict()
                    else:
                        msd = model.state_dict()
                    osd = optim.state_dict()
                    checkpoint = {
                        'model_states' : msd, 
                        'optim_states' : osd,
                        'epoch' : best_epoch,
                        'loss' : best_loss
                    }
                    torch.save(checkpoint, f'{save_dir}/model.pt')
                
                if epoch - best_epoch > 2.5 * patience:
                    break
            
            model_path = f'{save_dir}/model.pt'
            
            if self.post_eval is not None:
                self.post_eval(model_path, gpu=gpu)
            else:
                raise Exception('Default Eval. Not Implemented.')








