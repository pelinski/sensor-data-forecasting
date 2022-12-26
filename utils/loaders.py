import wandb
import torch
import yaml
import argparse
import pprint as pp

from models.lstm import CustomLSTM
from models.transformer import TransformerEncoder


def load_hyperparams():
    """Loads hyperparameters from a config yaml file.

    Returns:
        (dict): hyperparams
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="yaml config file", default=None, type=str)
    parser.add_argument("--model", help="model",
                        default="lstm", type=str)
    parser.add_argument("--num_sensors", help="number of sensors",
                        default=2, type=int)
    parser.add_argument("--dataset", help="dataset path",
                        default='dataset/data/test/RX0', type=str)
    parser.add_argument("--n_tgt_win", help="number of windows to be predicted",
                        default=1, type=int)
    parser.add_argument("--seq_len", help="maximum sequence length",
                        default=12, type=int)
    parser.add_argument("--batch_size", help="batch size",
                        default=32, type=int)
    parser.add_argument("--learning_rate",
                        help="learning rate", default=0.001, type=float)
    parser.add_argument(
        "--optimizer", help="optimizer algorithm", default='sgd', type=str)
    parser.add_argument(
        "--epochs", help="number of training epochs", default=1, type=int)
    parser.add_argument("--d_model", help="model dimension",
                        default=64, type=int)
    parser.add_argument("--dropout", help="dropout factor",
                        default=0.2, type=float)
    parser.add_argument(
        "--n_heads", help="number of heads for multihead attention", default=16, type=int)
    parser.add_argument(
        "--dim_feedforward", help="feed forward layer dimension", default=256, type=int)
    parser.add_argument(
        "--num_encoder_layers", help="number of encoder layers", default=7, type=int,)
    parser.add_argument("--embedding_size_src",
                        help="input embedding size", default=8, type=int,)
    parser.add_argument(
        "--embedding_size_tgt",
        help="output embedding size", default=8, type=int,)
    parser.add_argument(
        "--save_and_plot_period",
        help="save model and plot sample period in epochs", default=1, type=int,)
    parser.add_argument(
        "--plot_number",
        help="number of samples to plot", default=1, type=int,)
    parser.add_argument(
        "--load_model_epoch",
        help="load model at epoch", default=None, type=int,)
    parser.add_argument(
        "--load_model_path",
        help="wandb path to load model from", default=None, type=str,)

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            hp = yaml.safe_load(f)
    else:
        hp = {}

    hyperparams = {"model": hp["model"] if "model" in hp else args.model,
                   "num_sensors": hp["num_sensors"] if "num_sensors" in hp else args.num_sensors,
                   "dataset":  hp["dataset"] if "dataset" in hp else args.dataset,
                   "n_tgt_win": hp["n_tgt_win"] if "n_tgt_win" in hp else args.n_tgt_win,
                   "epochs": hp["epochs"] if "epochs" in hp else args.epochs,
                   "batch_size": hp["batch_size"] if "batch_size" in hp else args.batch_size,
                   "seq_len": hp["seq_len"] if "seq_len" in hp else args.seq_len,
                   "dropout": hp["dropout"] if "dropout" in hp else args.dropout,
                   "learning_rate": hp["learning_rate"] if "learning_rate" in hp else args.learning_rate,
                   "optimizer": hp["optimizer"] if "optimizer" in hp else args.optimizer,
                   "save_and_plot_period": hp["save_and_plot_period"] if "save_and_plot_period" in hp else args.save_and_plot_period,
                   "plot_number": hp["plot_number"] if "plot_number" in hp else args.plot_number,
                   "load_model_epoch": hp["load_model_epoch"] if "load_model_epoch" in hp else args.load_model_epoch,
                   "load_model_path": hp["load_model_path"] if "load_model_path" in hp else args.load_model_path,
                   "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')}
    if hyperparams["model"] == "transformer":
        hyperparams.update(
            {"embedding_size_src": hp["embedding_size_src"] if "embedding_size_src" in hp else args.embedding_size_src,
             "embedding_size_tgt": hp["embedding_size_tgt"] if "embedding_size_tgt" in hp else args.embedding_size_tgt,
             "num_heads": hp["num_heads"] if "num_heads" in hp else args.n_heads,
             "dim_feedforward": hp["dim_feedforward"] if "dim_feedforward" in hp else args.dim_feedforward,
             "num_encoder_layers": hp["num_encoder_layers"] if "num_encoder_layers" in hp else args.num_encoder_layers,
             "d_model": hp["d_model"] if "d_model" in hp else args.d_model,
             }
        )

    return dict(hyperparams)


def load_model(hyperparams):
    """ Creates models based on hyperparameters.

    Args:
        hyperparams (dict): dict containing hyperparameters

    Returns:
        model (torch.nn.Module): Model based on hyperparameters
        epoch (int): Epoch number
    """

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if hyperparams["model"] == "lstm":
        model = CustomLSTM(hidden_size=hyperparams["num_sensors"], input_size=hyperparams["num_sensors"], out_size=hyperparams["n_tgt_win"]*hyperparams["seq_len"], **hyperparams).to(
            device=device, non_blocking=True)
    elif hyperparams["model"] == "transformer":
        model = TransformerEncoder(out_size=hyperparams["n_tgt_win"]*hyperparams["seq_len"],
                                   **hyperparams).to(device=device, non_blocking=True)
    else:
        model = None

    if hyperparams["load_model_epoch"] and hyperparams["load_model_path"]:
        model_file = wandb.restore("run_{}_epoch_{}.model".format(hyperparams["load_model_path"].split(
            "/")[-1], hyperparams["load_model_epoch"]), run_path=hyperparams["load_model_path"])
        checkpoint = torch.load(
            model_file.name, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint['epoch']
    else:
        epoch = 0

    return model, epoch+1


def load_optimizer(model, hyperparams):
    """Returns optimizer based on hyperparameters

    Args:
        model (torch.nn.Module): Torch model
        hyperparams (dict): Dict containing hyperparameters

    Returns:
        optimizer (torch.optim): Optimizer
    """

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if hyperparams["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyperparams["learning_rate"])
    elif hyperparams["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=hyperparams["learning_rate"])
    else:
        optimizer = None

    if hyperparams["load_model_epoch"] and hyperparams["load_model_path"]:
        model_file = wandb.restore("run_{}_epoch_{}.model".format(
            hyperparams["load_model_path"].split(
                "/")[-1], hyperparams["load_model_epoch"]), run_path=hyperparams["load_model_path"])
        checkpoint = torch.load(
            model_file.name, map_location=torch.device(device))

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer
