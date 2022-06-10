import argparse
import logging
import os
import shutil
import sys

import torch.utils.data as td
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from neural_network.dataset import create_datasets
from neural_network.model import build_model
from neural_network.utils import conf
from neural_network.utils.utils import *

logger = logging.getLogger(__name__)


def do_evaluation_overfit(model, data, device, loss_fn, metrics_fn):
    """Evaluate the model on a single datapoint provided to the function and according to the given loss and metrics"""
    model.eval()
    results = {}

    data = batch_to_device(data, device, non_blocking=True)
    with torch.no_grad():
        pred = model(data)
        losses = loss_fn(pred, data)
        metrics = metrics_fn(pred, data)
        del pred, data

    numbers = {**metrics, **{'loss/' + k: v for k, v in losses.items()}}
    for k, v in numbers.items():
        if k not in results:
            results[k] = AverageMetric()
        results[k].update(v)

    results = {k: results[k].compute() for k in results}
    return results


def do_evaluation(model, loader, device, loss_fn, metrics_fn):
    """Evaluate the model on the dataset provided by the loader and according to the given loss and metrics"""
    model.eval()
    results = {}
    for data in loader:
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses = loss_fn(pred, data)
            metrics = metrics_fn(pred, data)
            del pred, data

        numbers = {**metrics, **{'loss/' + k: v for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()

            results[k].update(v)

    results = {k: results[k].compute() for k in results}
    return results


def overfit_model(dataroot, model, config, output_path):
    """Train the model on the whole dataset"""

    set_seed(config["seed"])
    writer = SummaryWriter(log_dir=str(output_path))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device {device}')

    use_wandb = True if config['key'] is not None else False

    # Loading the fragment data
    train, test = create_datasets(dataroot, conf=config)
    print(f"Train size: {train.__len__()}\nTest size:{test.__len__()}")

    # create a data loader for train and test sets
    train_dl = td.DataLoader(
        train,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f'Training loader has {len(train_dl)} batches')
    logger.info(f'Overfitting on first batch')

    loss_fn, metrics_fn = model.loss, model.metrics
    model = model.to(device)

    logger.info(f'Model: \n{model}')
    torch.backends.cudnn.benchmark = True

    optimizer_fn = {'sgd': torch.optim.SGD,
                    'adam': torch.optim.Adam,
                    'rmsprop': torch.optim.RMSprop}[config["optimizer"]]

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    optimizer = optimizer_fn([p for n, p in params], lr=config["lr"])

    def lr_fn(it):
        if config["lr_schedule"]["type"] is None:
            return 1
        if config["lr_schedule"]["type"] == 'exp':
            gam = 10 ** (-1 / config["lr_schedule"]["exp_div_10"])
            return 1 if it < config["lr_schedule"]["start"] else gam
        else:
            raise ValueError(config["lr_schedule"]["type"])

    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
    logger.info(f'Starting overfitting with configuration:\n{config}')

    epoch = 0
    best_eval = 10000
    while epoch < config["epochs"]:
        logger.info(f'Starting epoch {epoch}')
        set_seed(config["seed"] + epoch)

        # do overfitting on first batch in train_dl
        data = [i for i in train_dl][0]

        # train, evaluate, backprop, update lr
        model.train()
        optimizer.zero_grad()
        data = batch_to_device(data, device, non_blocking=True)
        pred = model(data)
        losses = loss_fn(pred, data)
        loss = torch.mean(losses['total'])
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # evaluate on test set
        results = do_evaluation_overfit(model, data, device, loss_fn, metrics_fn)
        str_results = [f'{k}: {v:.3E}' for k, v in results.items()]

        # log to wandb
        if use_wandb:
            wandb.log({'match_recall': results['match_recall']})
            wandb.log({'match_precision': results['match_precision']})
            wandb.log({'loss/total': results['loss/total']})
            wandb.log({'lr': optimizer.param_groups[0]['lr']})

        torch.cuda.empty_cache()
        logging.info(f"Overfitting Epoch: {epoch}")
        epoch += 1
        del pred, data, loss, losses

    logger.info(f'Overfitting finished')
    writer.close()


def train_model(dataroot, model, config, output_path):
    """Train the model on the whole dataset"""

    set_seed(config["seed"])
    writer = SummaryWriter(log_dir=str(output_path))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device {device}')

    use_wandb = True if config['key'] is not None else False

    # Loading the fragment data
    train, test = create_datasets(dataroot, conf=config)
    print(f"Train size: {train.__len__()}\nTest size:{test.__len__()}")

    # create a data loader for train and test sets
    train_dl = td.DataLoader(
        train,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_dl = td.DataLoader(
        test,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    logger.info(f'Training loader has {len(train_dl)} batches')
    logger.info(f'Validation loader has {len(test_dl)} batches')

    loss_fn, metrics_fn = model.loss, model.metrics
    model = model.to(device)

    logger.info(f'Model: \n{model}')
    torch.backends.cudnn.benchmark = True

    optimizer_fn = {'sgd': torch.optim.SGD,
                    'adam': torch.optim.Adam,
                    'rmsprop': torch.optim.RMSprop}[config["optimizer"]]

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    optimizer = optimizer_fn([p for n, p in params], lr=config["lr"])

    def lr_fn(it):
        if config["lr_schedule"]["type"] is None:
            return 1
        if config["lr_schedule"]["type"] == 'exp':
            gam = 10 ** (-1 / config["lr_schedule"]["exp_div_10"])
            return 1 if it < config["lr_schedule"]["start"] else gam
        else:
            raise ValueError(config["lr_schedule"]["type"])

    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
    logger.info(f'Starting training with configuration:\n{config}')

    # initiate train values
    epoch = 0
    best_eval = 0

    while epoch < config["epochs"]:

        logger.info(f'Starting epoch {epoch}')
        set_seed(config["seed"] + epoch)

        for it, data in enumerate(tqdm(train_dl, desc=f'Training epoch {epoch}')):
            tot_it = len(train_dl) * epoch + it

            model.train()
            optimizer.zero_grad()
            data = batch_to_device(data, device, non_blocking=True)
            pred = model(data)
            losses = loss_fn(pred, data)
            loss = torch.mean(losses['total'])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if it == len(train_dl) - 1:
                for k in sorted(losses.keys()):
                    losses[k] = torch.mean(losses[k]).item()
                str_losses = []

                for k, v in losses.items():
                    str_losses.append(f'{k} {v:.3E}')
                    if use_wandb:
                        wandb.log({f'{k}': v})

                metr = metrics_fn(pred, data)
                prec = np.mean([p.item() for p in metr['match_precision']])
                rec = np.mean([p.item() for p in metr['match_recall']])
                prec = 0 if prec != prec else prec
                rec = 0  if rec != rec else rec


                logger.info('[Epoch {}] train loss {{{}}}'.format(epoch, ', '.join(str_losses)))
                logger.info('[Epoch {}] train precision: {}, recall: {}'.format(epoch, prec, rec))

                for k, v in losses.items():
                    writer.add_scalar('training/' + k, v, tot_it)

                writer.add_scalar('training/lr', optimizer.param_groups[0]['lr'], tot_it)
                if use_wandb:
                    wandb.log({'precision_train': prec})
                    wandb.log({'recall_train': rec})
                    wandb.log({'lr': optimizer.param_groups[0]['lr']})

            if it == len(train_dl) - 1:
                results = do_evaluation(model, test_dl, device, loss_fn, metrics_fn)

                str_results = [f'{k}: {v:.3E}' for k, v in results.items()]
                if use_wandb:
                    wandb.log({'match_recall': results['match_recall']})
                    wandb.log({'match_precision': results['match_precision']})
                    wandb.log({'loss_test': results['loss/total']})
                    plot_matching_vector(data, pred)

                logging.info(f'[Validation] {{{", ".join(str_results)}}}')
                for k, v in results.items():
                    writer.add_scalar('val/' + k, v, tot_it)
                torch.cuda.empty_cache()  # should be cleared at the first iter

            del pred, data, loss, losses

        # save model every 100th epoch
        if epoch % 100 == 0:

            cp_name = 'checkpoint_{}'.format(epoch)
            logger.info('Saving checkpoint {}'.format(cp_name))
            cp_path = str(output_path + "/" + (cp_name + '.tar'))
            torch.save(model.state_dict(), cp_path)

        if results['match_precision'] < best_eval:
            best_eval = results['match_precision']
            logger.info(f'New best checkpoint: precision = {best_eval}')
            shutil.copy(cp_path, str(output_path + "/" + 'checkpoint_best.tar'))
        
        epoch += 1

    logger.info(f'Finished training.')

    writer.close()


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None)
    args = parser.parse_intermixed_args()

    model_conf, train_conf, data_conf, wandb_conf = conf.model_conf, conf.train_conf, conf.data_conf, conf.wandb_conf
    here = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    if args.path == None:
        root = os.path.join(here, '..', 'object_fracturing', 'data')
    else:
        root = args.path

    np.set_printoptions(threshold=sys.maxsize)

    config = {**model_conf, **train_conf, **data_conf, **wandb_conf}
    model = build_model(config['weights'], config)

    # wandb login
    if wandb_conf['key'] is not None:
        wandb.login(key=wandb_conf['key'], relogin=False)
        wandb.init(project=wandb_conf['project'],
                   entity=wandb_conf['entity'],
                   config=config,
                   settings=wandb.Settings(start_method='thread'))
        config = wandb.config
        wandb.watch(model)
        output_path = os.path.join(here, '_'.join(['model_output', wandb.run.name]))
        name_weights = os.path.join(here, 'weights', f'model_weights_{wandb.run.name}.pth')
    else:
        output_path = os.path.join(here,'model_output')
        name_weights = os.path.join(here, 'weights', 'model_weights.pth')
    shutil.rmtree(output_path, ignore_errors=True)

    train_model(root, model, config, output_path)
    torch.save(model.state_dict(), name_weights)
