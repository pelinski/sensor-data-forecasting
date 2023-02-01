import torch
import numpy as np
import argparse
from utils.loaders import load_model, load_hyperparams_from_wandb
from tinynn.converter import TFLiteConverter
import os
import pprint as pp


def export_to_tflite(model, dummy_input, converted_model_path):
    with torch.no_grad():
        model.cpu()
        model.eval()
        converter = TFLiteConverter(model, dummy_input, converted_model_path)
        converter.convert()


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--run_path", help="wandb run path", default=None, type=str)
    parser.add_argument(
        "-e", "--epoch", help="epoch to load", default=None, type=int)
    args = parser.parse_args()

    # load model params from wandb
    trained_models_dir = "trained_models"
    run_id = args.run_path.split("/")[-1]
    params_filename = "{}-params.pk".format(run_id)
    params = load_hyperparams_from_wandb(
        params_filename, args.run_path, root="trained_models/")
    os.remove("{}/{}".format(trained_models_dir, params_filename))
    pp.pprint(params, sort_dicts=False)

    # load model from wandb
    params["run_path"] = args.run_path
    params["load_model_epoch"] = args.epoch
    model, epoch = load_model(params, root=trained_models_dir)
    converted_model_path = "{}/run_{}_epoch_{}.tflite".format(
        trained_models_dir, run_id, epoch-1)

    # total params
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {}".format(total_params))
    
    # dummy input
    dummy_input = torch.randn(
        params["batch_size"], params["seq_len"], params["num_sensors"])

    # convert to tflite
    export_to_tflite(model, dummy_input, converted_model_path)
