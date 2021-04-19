import higher
import torch
import numpy as np

def validate(model, batch, loss_fn, critic):

    model.eval()

    flair = batch['FL']
    t1 = batch['T1']
    y = batch['label']

    metrics = dict()

    scoresT1 = model(t1)
    lossT1 = loss_fn(scoresT1, y)

    for k,v in critic(scoresT1, y).items():
        metrics[f'T1-{k}'] = v
    
    metrics[f'T1-loss'] = lossT1.item()

    scoresFL = model(flair)
    lossFL = loss_fn(scoresFL, y)

    for k,v in critic(scoresFL, y).items():
        metrics[f'FL-{k}'] = v
    
    metrics[f'FL-loss'] = lossFL.item()
    metrics['loss'] = .5 * (lossT1 + lossFL).item()
    
    return metrics

def _extract(batch, agg=True):
    
    if agg:

        fl = [x.unsqueeze(0) for x,k in zip(batch['data'], 
            batch['kind']) if k == 'FL']
        num_fl = len(fl)
        fl = torch.cat(fl, dim=0) if len(fl) > 0 else None

        t1 = [x.unsqueeze(0) for x,k in zip(batch['data'], 
            batch['kind']) if k == 'T1']
        num_t1 = len(t1)
        t1 = torch.cat(t1, dim=0) if len(t1) > 0 else None

        fl_y = [x.unsqueeze(0) for x,k in zip(batch['label'], 
            batch['kind']) if k == 'FL']
        fl_y = torch.cat(fl_y, dim=0) if len(fl_y) > 0 else None

        t1_y = [x.unsqueeze(0) for x,k in zip(batch['label'], 
            batch['kind']) if k == 'T1']
        t1_y = torch.cat(t1_y, dim=0) if len(t1_y) > 0 else None
    
    else:

        fl = batch['FL']
        num_fl = fl.size(0)
        t1 = batch['T1']
        num_t1 = t1.size(0)
        fl_y = batch['FL_label']
        t1_y = batch['T1_label']
    
    return {'FL' : fl, 'T1' : t1, 'FL_label' : fl_y,
        'T1_label' : t1_y, 'num_FL' : num_fl, 'num_T1' : num_t1}

def _compute_loss(model, batch, loss_fn, train=False):

    if train:
        model.train()
    else:
        model.eval()

    x = []
    for k in ['FL', 'T1']:
        if batch[f'num_{k}'] > 0:
            x.append(batch[k])
    
    scores = model(torch.cat(x, dim=0))

    fl_scores = scores[:batch['num_FL']]
    t1_scores = scores[batch['num_FL']:]

    lossFL = loss_fn(fl_scores, batch['FL_label']) \
        if batch['num_FL'] > 0 else 0
    lossT1 = loss_fn(t1_scores, batch['T1_label']) \
        if batch['num_T1'] > 0 else 0
    
    return {'FL_loss' : lossFL, 'T1_loss' : lossT1}

def standard_update(model, batch, optim, loss_fn,
    agg=True, proportional=False, lamb=None):

    metrics = dict()
    optim.zero_grad()

    batch = _extract(batch, agg=agg)
    
    losses = _compute_loss(model, batch, loss_fn, train=True)
    lossFL = losses['FL_loss']
    lossT1 = losses['T1_loss']
    
    if proportional:

        total = batch['num_FL']+batch['num_T1']
        lossFL *= batch['num_FL'] / total
        lossT1 *= batch['num_T1'] / total
    
    elif lamb is not None:

        lossFL *= lamb
        lossT1 *= (1-lamb)
    else:
        # assume lambda is 0.50
        lossFL *= 0.5
        lossT1 *= 0.5
    
    loss = lossFL + lossT1

    loss.backward()
    optim.step()

    metrics['FL-contrib'] = lossFL.item() \
        if torch.is_tensor(lossFL) else lossFL
    metrics['T1-contrib'] = lossT1.item() \
        if torch.is_tensor(lossT1) else lossT1
    metrics['loss'] = loss.item()

    return metrics

def _hypothetical_step(model, optim, loss_fn, train, test):

    with higher.innerloop_ctx(model, optim,
        track_higher_grads=False, copy_initial_weights=True
        ) as (fmodel, foptim):

        fmodel.train()
        x, y = train
        scores = fmodel(x)
        loss = loss_fn(scores, y)
        foptim.step(loss)
        
        holdout_loss = []
        fmodel.eval()
        # single-pass not necessary since eval
        for x, y in test:

            scores = fmodel(x)
            holdout_loss.append(loss_fn(scores, y))

    # .5 flair  + .5 t1
    return sum(holdout_loss) / len(holdout_loss)

def _split(batch):

    split = dict()
    split['train'] = dict()
    split['test'] = dict()

    # use roughly half of batch for train and half for validation
    n = min(batch['num_FL'], batch['num_T1'])
    meta_train = n // 2
    if meta_train < 1:
        # return None lambda will not be updated in this case
        return None

    for kind in ['FL', 'T1']:

        for k in [kind, f'{kind}_label']:
            split['train'][k] = batch[k][:meta_train]
        
        split['train'][f'num_{kind}'] = meta_train
        
        for k in [kind, f'{kind}_label']:
            split['test'][k] = batch[k][meta_train:]
    
    return split
    
def moving_update(model, batch, optim, loss_fn, agg=False,
    style=None, hypers=None):

    assert hypers is not None and style is not None

    metrics = dict()
    optim.zero_grad()

    batch = _extract(batch, agg=agg)
    split = _split(batch)

    if split is not None:

        fl_train = (split['train']['FL'], split['train']['FL_label'])
        t1_train = (split['train']['T1'], split['train']['T1_label'])
        test = [(split['test'][k], split['test'][f'{k}_label'])
            for k in ['FL', 'T1']]

        hypot_loss_FL = _hypothetical_step(model, optim, loss_fn,
            fl_train, test)
        hypot_loss_T1 = _hypothetical_step(model, optim, loss_fn,
            t1_train, test)
        
        if style == 'simple':
            _simple_lambda_update(hypot_loss_FL, 
                hypot_loss_T1, hypers)
        elif style == 'map':
            _map_lambda_update(hypot_loss_FL, 
                hypot_loss_T1, hypers)
        else:
            raise NotImplementedError(f' for name {style}')

    lamb = hypers['lambda']
    losses = _compute_loss(model, batch, loss_fn, train=True)
    lossFL = lamb * losses['FL_loss']
    lossT1 = (1 - lamb) * losses['T1_loss']
    loss = lossFL + lossT1

    loss.backward()
    optim.step()

    metrics['lambda'] = lamb.item() \
        if torch.is_tensor(lamb) else lamb
    metrics['FL-contrib'] = lossFL.item() \
        if torch.is_tensor(lossFL) else lossFL
    metrics['T1-contrib'] = lossT1.item() \
        if torch.is_tensor(lossT1) else lossT1
    metrics['loss'] = loss.item()

    return metrics

def init_params(params):

    if params['style'] == 'simple':

        params['hypers']['lambda'] = 0.5
        params['hypers']['eta'] = 0.1

    elif params['style'] == 'map':

        params['hypers']['alpha'] = 5
        params['hypers']['beta'] = 5
        params['hypers']['history'] = []
        params['hypers']['lambda'] = 0.5

def _simple_lambda_update(hypot_loss_FL, hypot_loss_T1, hypers):

    diff = hypot_loss_FL.item() - hypot_loss_T1.item()

    if hypers['direction'] == '>':
        diff = diff
    elif hypers['direction'] == '<':
        diff = -diff
    else:
        raise NotImplementedError('Must specify the bernoulli var.')

    perc_diff = diff / abs(hypot_loss_FL.item())
    
    hypers['lambda'] = hypers['lambda'] + hypers['eta'] * perc_diff
    hypers['lambda'] = max(0, min(1, hypers['lambda']))

    return None

def _map_lambda_update(hypot_loss_FL, hypot_loss_T1, hypers):
    
    if hypers['direction'] == '>':
        xt = int(hypot_loss_FL.item() > hypot_loss_T1.item())
    elif hypers['direction'] == '<':
        xt = int(hypot_loss_FL.item() < hypot_loss_T1.item())
    else:
        raise NotImplementedError('Must specify the bernoulli var.')
    hypers['history'].append(xt)
    if len(hypers['history']) > hypers['window_size']:
        hypers['history'] = hypers['history'][1:]
    sumxt = sum(hypers['history'])
    n = len(hypers['history'])
    numer = hypers['alpha'] + sumxt - 1
    denom = hypers['alpha'] + hypers['beta'] + n - 2
    hypers['lambda'] = numer / denom

    return None

PROCEDURES = {
    'validate' : validate,
    'standard_update' : standard_update,
    'moving_update' : moving_update,
    'moving_init' : init_params
}
