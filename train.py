import json
import os.path as osp
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph, knn_graph
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from tqdm import tqdm
import argparse
import functools
import re
import time
import yaml
from glob import glob
import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
import warnings
warnings.simplefilter('ignore')
from time import strftime, gmtime

'''

Change from DeepMETv2

1. Change x_cont, x_cat, etaphi in train() in accordance with the change of training inputs 
2. Add n_features_cont, n_features_cat to keep track of these numbers here and there, i.e. when building a model these numbers go into arguments
3. Remove the resolution-MET plotting part in evaluate, as L1 doesn't have bunch of METs that DeepMETv2 has access to
4. Add input scaling to [0,1] (norm)
5. Add weight_decay to the optimizer, remove patience from the scheduler 

'''


parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--data', default='data_ttbar',
                    help="Dataset root folder (PyG reads its processed/ subfolder)")
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")
parser.add_argument('--cfg', default='configs/config1.yaml',
                    help="Path to config YAML file")


def train(model, device, optimizer, scheduler, loss_fn, dataloader):
    model.train()
    
    loss_avg_arr = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader), disable=True) as t:
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)

            x_cont = data.x[:,:n_features_cont]       # include puppi
            #x_cont = data.x[:,:(n_features_cont-1)]  # remove puppi
            x_cat = data.x[:,n_features_cont:].long()

            #phi = torch.atan2(data.x[:,2], data.x[:,1])   # atan2(py, px)
            etaphi = torch.cat([data.x[:,3][:,None], data.x[:,4][:,None]], dim=1)

            # NB: there is a problem right now for comparing hits at the +/- pi boundary
            edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, loop=False, max_num_neighbors=255)  # turn off self-loop
            result = model(x_cont, x_cat, edge_index, data.batch)
            
            loss = loss_fn(result, data.x, data.y, data.batch, scale_momentum=scale_momentum)
            loss.backward()
            optimizer.step()
            
            # update the average loss
            loss_avg_arr.append(loss.item())
            
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    
    scheduler.step(np.mean(loss_avg_arr))

    return np.mean(loss_avg_arr)


if __name__ == '__main__':
    args = parser.parse_args()

    # load config
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)

    n_features_cont   = int(config['N_FEATURES_CONT'])
    n_features_cat    = int(config['N_FEATURES_CAT'])
    scale_momentum    = int(config['SCALE_MOMENTUM'])   # scaling factor of pT, px, py (hence the target MET)
    epochs            = int(config['N_EPOCHS'])
    deltaR            = float(config['DELTA_R'])
    hidden_dim        = int(config['HIDDEN_DIM'])
    conv_depth        = int(config['CONV_DEPTH'])
    activation        = config.get('ACTIVATION_FUNCTION', 'relu')
    loss_type         = config.get('LOSS_TYPE', 'response_tune')
    loss_c            = float(config.get('LOSS_C', 5000.))
    loss_pt_threshold = float(config.get('LOSS_PT_THRESHOLD', 50.))
    learning_rate     = float(config.get('LEARNING_RATE', 1e-5))
    max_lr            = float(config.get('MAX_LR', 1e-4))
    weight_decay      = float(config.get('WEIGHT_DECAY', 0.001))
    batch_size        = int(config.get('BATCH_SIZE', 6))
    validation_split  = float(config.get('VALIDATION_SPLIT', 0.2))
    use_edge_features = bool(config.get('USE_EDGE_FEATURES', False))

    # load data
    dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ['PWD'],args.data),
                                               batch_size=batch_size,
                                               validation_split=validation_split)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    print('Training dataloader: {}, Test dataloader: {}'.format(len(train_dl), len(test_dl)))

    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # norm for the input data
    norm = torch.tensor([1./scale_momentum, 1./scale_momentum, 1./scale_momentum, 1., 1., 1.]).to(device)   # pt, px, py: scale by scale_momentum

    # model
    model = net.Net(n_features_cont, n_features_cat, norm, hidden_dim=hidden_dim, conv_depth=conv_depth, activation=activation, use_edge_features=use_edge_features).to(device) #include puppi
    #model = net.Net(n_features_cont-1, n_features_cat, norm, hidden_dim=hidden_dim, conv_depth=conv_depth, activation=activation, use_edge_features=use_edge_features).to(device) #remove puppi

    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate, weight_decay=weight_decay)

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.1)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = learning_rate, max_lr = max_lr, cycle_momentum=False)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=0.05)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500, threshold=0.05)

    first_epoch = 0
    best_validation_loss = 10e7
    deltaR_dz = 0.3 # not used

    # loss function
    if loss_type == 'response_tune':
        loss_fn = functools.partial(net.loss_fn_response_tune, c=loss_c, pt_threshold=loss_pt_threshold)
    else:
        loss_fn = net.loss_fn

    metrics = net.metrics

    # output dir: one folder per run, auto-named "<config_id>_run<N>" under the
    # --ckpts root, mirroring TransforMET's results/<config>_runN layout.
    # If resuming (--restore_file), --ckpts is treated as the existing run folder.
    config_id = osp.splitext(osp.basename(args.cfg))[0]
    models_root = osp.join(os.environ['PWD'], args.ckpts)
    if args.restore_file is not None:
        model_dir = models_root
    else:
        os.makedirs(models_root, exist_ok=True)
        existing_runs = glob(osp.join(models_root, '{}_run*'.format(config_id)))
        run_indices = []
        for path in existing_runs:
            match = re.search(r'_run(\d+)$', osp.basename(path))
            if match:
                run_indices.append(int(match.group(1)))
        run_idx = max(run_indices) + 1 if run_indices else 0
        model_dir = osp.join(models_root, '{}_run{}'.format(config_id, run_idx))
    os.makedirs(model_dir, exist_ok=True)
    print('Output directory:', model_dir)

    # save a snapshot of the config alongside the checkpoints
    with open(osp.join(model_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    loss_log = open(model_dir+'/loss.log', 'w')
    loss_log.write('# loss log for training starting in '+strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
    loss_log.write('epoch, loss, val_loss\n')
    loss_log.flush()

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
        ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Restarting training from epoch',first_epoch)
        with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']
            
    # Train
    n_steps = len(train_dl)
    for epoch in range(first_epoch+1, epochs):
        print('Epoch {}/{}'.format(epoch, epochs-1), flush=True)
        epoch_start = time.time()

        train_loss = train(model, device, optimizer, scheduler, loss_fn, train_dl)

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'optim_dict': optimizer.state_dict(),
                                'sched_dict': scheduler.state_dict()},
                                is_best=False,
                                checkpoint=model_dir)

        # save model
        # m = torch.jit.script(model)
        # torch.jit.save(m, f'{model_dir}/MODELS/scripted_model_epoch{epoch}.pt')
        # torch.save(model, f'{model_dir}/MODELS/model_epoch{epoch}.pt')

        # Evaluate for one epoch on validation set
        test_metrics, resolutions, MET_arr = evaluate(model, device, loss_fn, test_dl, metrics, deltaR, deltaR_dz, model_dir, epoch, n_features_cont = n_features_cont, scale_momentum = scale_momentum, save_METarr = True)
        # test_metrics, resolution_hists, MET_arr = evaluate(model, device, loss_fn, test_dl, metrics, deltaR, deltaR_dz, model_dir, epoch, save_METarr = False)

        validation_loss = test_metrics['loss']
        elapsed = time.time() - epoch_start
        last_lr = scheduler.state_dict().get('_last_lr', [float('nan')])[0]
        print('{n}/{n} - {t:.0f}s - {ms:.0f}ms/step - loss: {tl:.4f} - val_loss: {vl:.4f} - learning_rate: {lr:.4g}'.format(
            n=n_steps, t=elapsed, ms=1000.0*elapsed/max(n_steps, 1), tl=train_loss, vl=validation_loss, lr=last_lr), flush=True)
        loss_log.write('%d,%.8f,%.8f\n'%(epoch, train_loss, validation_loss))
        loss_log.flush()
        is_best = (validation_loss<=best_validation_loss)

        # If best_eval, best_save_path
        if is_best: 
            print('Found new best loss!') 
            best_validation_loss=validation_loss

            # Save weights
            utils.save_checkpoint({'epoch': epoch,
                                    'state_dict': model.state_dict(),
                                    'optim_dict': optimizer.state_dict(),
                                    'sched_dict': scheduler.state_dict()},
                                    is_best=True,
                                    checkpoint=model_dir)

            # Save best val metrics in a json file in the model directory
            utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_best.json'))
            utils.save(resolutions, osp.join(model_dir, 'best.resolutions'))

        utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_last.json'))
        utils.save(resolutions, osp.join(model_dir, 'last.resolutions'))

    loss_log.close()
