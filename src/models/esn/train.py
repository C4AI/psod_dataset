import os
import pandas as pd
import torch
import traceback
import time
import wandb
import numpy as np
import sys
import argparse
import json

sys.path.append("src")

from model import TCLiESN
from dataset import context_train_dataset, test_batcher


def train_esn():
    wandb.login(key=esn_config["wandb_api_key"])  # log into Weights and Biases

    # Configure Sweep hyperparameter ranges and search algorithm in a dict

    sweep_config = {
        'method': esn_config['hyperparameter_opt_method'],
        'name': 'sweep',
        'metric': {
            'name': 'mean_rmse_val',
            'goal': 'minimize'
        },
        'parameters': esn_config['sweep_parameters']
    }

    try:
        if esn_config['sweep_id'] is None:  # Create new sweep
            esn_config['sweep_id'] = wandb.sweep(sweep_config, project=esn_config['project_name'],
                                                 entity=esn_config['username'])

        wandb.agent(sweep_id=esn_config['sweep_id'], project=esn_config['project_name'], function=esn_train,
                    count=esn_config['steps'])

    except Exception:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)


def esn_train(config=None):
    try:

        with wandb.init(config=config, project=esn_config['project_name'], entity=esn_config['username']):

            # Create folder to save the weights

            os.makedirs(esn_config['save_path'], exist_ok=True)

            dict_train = TCLiESN.config_2_dict(**wandb.config)  # Receives the hyperparameter dict from wandb
            dict_train['seed'] = esn_config['seed']
            dict_train['sparse'] = True
            dict_train['torch_type'] = dataset_config['dtype']
            dict_train["device"] = dataset_config['device']

            # Split data into training and validation
            train_dl = context_train_dataset(**dataset_config)  # Dataloader
            train_ds, validate_ds = train_dl.split_batched_train_val(
                val_period_percentage=esn_config['validation_percentage'],  # % of data for validation
                train_stride=esn_config["training_stride"],  # The stride of training batches
                val_stride=esn_config["validation_stride"])  # The stride of validation batches

            dict_train['input_dim'], dict_train['output_dim'], dict_train['out_maps'], dict_train[
                'input_map'] = train_dl.get_dimensions()

            train_columns, pred_columns = train_dl.get_columns_names()

            start = time.time()
            # create a Time Continuous ESN
            esn = TCLiESN(**dict_train)
            # reset training state
            esn.reset()
            # train the esn in the training dataset
            esn.train_epoch(train_ds)
            # compute trained weights
            esn.train_finalize()
            # Calculate loss in the validation dataset
            inputs, predictions, losses, rmses = esn.predict_batches(validate_ds,
                                                                     forecast_horizon=esn_config["forecast_horizon"],
                                                                     warmup=esn_config["warmup"],
                                                                     return_rmse=True)
            losses = np.array(losses)
            rmses = np.array(rmses)

            mean_loss = np.nanmean(losses, 0)
            mean_std = np.nanstd(losses, 0)
            mean_rmse = np.nanmean(rmses, 0)
            std_rmse = np.nanstd(rmses, 0)
            mean_mean_ioa = np.nanmean(mean_loss)
            mean_mean_rmse = np.nanmean(mean_rmse)

            # Log metrics into the wandb portal
            log_metrics = dict()
            for idx, column in enumerate(pred_columns):
                log_metrics['ioa_val_' + column] = mean_loss[idx]
                log_metrics['rmse_val_' + column] = mean_rmse[idx]
                log_metrics['ioa_std_' + column] = mean_std[idx]
                log_metrics['rmse_std_' + column] = std_rmse[idx]
            log_metrics['mean_ioa_val'] = mean_mean_ioa
            log_metrics['mean_rmse_val'] = mean_mean_rmse
            wandb.log(log_metrics)

            # Save run trained weights
            os.makedirs(esn_config['save_path'] + '/' + wandb.run.id + '/', exist_ok=True)
            esn.save_weights(esn_config['save_path'] + '/' + wandb.run.id + '/' + esn_config['save_name'] + '_' + wandb.run.id)
            with open(esn_config['save_path'] + '/' + wandb.run.id + '/' + esn_config['save_name'] + '_dict_train', 'w', encoding='utf-8') as f:
                try:
                    json.dump(dict(wandb.config), f, ensure_ascii=False, indent=4)
                except:
                    print('Error saving dict_train')

            print('mean losses val:' + str(mean_mean_ioa) + '+-' + str(mean_std))

            end = time.time()
            print("Finished one run! Total time: " + str(end - start))
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)


def load_dicts(arg):

    # Prepare esn dict for use
    ec = json.load(open(arg.esn_config, 'r'))

    ec['save_path'] = arg.save_path
    ec['save_name'] = arg.save_name
    ec['wandb_api_key'] = arg.wandb_api_key
    ec['warmup'] = pd.Timedelta(ec['warmup'], unit='hours')
    ec['forecast_horizon'] = pd.Timedelta(ec['forecast_horizon'], unit='hours')
    ec['training_stride'] = pd.Timedelta(ec['training_stride'], unit='hours')
    ec['validation_stride'] = pd.Timedelta(ec['validation_stride'], unit='hours')
    ec['device'] = torch.device(ec['device'])
    if ec['dtype'] == 'float16':
        ec['dtype'] = torch.float16
    elif ec['dtype'] == 'float32':
        ec['dtype'] = torch.float32
    elif ec['dtype'] == 'float64':
        ec['dtype'] = torch.float64

    wb = json.load(open(arg.wandb_config, 'r'))

    ec.update(wb)

    # Prepare dataset dict for use
    dc = json.load(open(arg.dataset_config, 'r'))
    dc['dtype'] = ec['dtype']
    dc['device'] = ec['device']
    dc['warmup'] = ec['warmup']
    dc['forecast_horizon'] = ec['forecast_horizon']
    dc['timeseries'] = tuple(dc['timeseries'])
    for ts in dc['timeseries']:
        ts['path'] = dc['folder_train'] + ts['filename']

    return ec, dc


if __name__ == '__main__':

    # The dataset_config is a json containing the time-series that will be used:
    #   "timeseries" : tuple of dicts, where each entry is one time series
    #   "input_datasets" : array of strings with the names of the datasets that will be used as input
    #   "target_datasets" : array of strings with the names of the datasets that will be the target
    # The array timeseries has the following parameters:
    #   "filename" : file name of the parquet dataset
    #   "forecast" : boolean indicating if the time series will is from a forecast
    #   "is_predicted" : boolean that indicates to the dataloader if the time series will be predicted by the model
    #   "transformations" : list of transformations that can be applied to the data (eg: lowpass filtering, z-score)
    #   "description" : description of the dataset

    # The esn_config is a json containing the parameters of the ESN model and WANDB configuration:
    #   "warmup": ESN warmup in hours
    #   "forecast_horizon": ESN forecast duration in hours
    #   "seed":  Seed for pytorch and numpy
    #   "training_stride": Stride for training sequential batches in hours
    #   "validation_stride": Stride for validation sequential batches in hours
    #   "validation_percentage": Percentage of the dataset that will be used for validation

    # The wandb_config is a json containing the WANDB configuration:
    #   "username": wandb username
    #   "project_name": wandb project name
    #   "sweep_id": wandb sweep, if null create a new sweep
    #   "steps": wandb sweep steps
    #   "hyperparameter_opt_method": Hyperparameter optimization method
    #   "sweep_parameters" : esn hyperparameters to be optimized

    parser = argparse.ArgumentParser(prog='esn_santos', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_config', help='Json with the dataset configuration.', required=True)
    parser.add_argument('--esn_config', help='Json with the ESN and Wandb configuration.', required=True)
    parser.add_argument('--wandb_config', help='Json with the WandB configuration', required=True)
    parser.add_argument('--wandb_api_key', help='WandB api key', required=True)
    parser.add_argument('--save_path', help='Path for saving esn output/trained weights', required=True)
    parser.add_argument('--save_name', help='Name for saving esn output/trained weights', default='esn')

    if len(sys.argv) == 1:
        print('\n')
        print(parser.print_help())
        sys.exit(-1)

    args = parser.parse_args()

    esn_config, dataset_config = load_dicts(args)

    train_esn()
