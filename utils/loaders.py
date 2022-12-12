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
                        default='dataset/data/chaos-bells-2/processed/RX0', type=str)
    parser.add_argument("--seq_len", help="maximum sequence length",
                        default=12, type=int)
    parser.add_argument("--batch_size", help="batch size",
                        default=32, type=int)
    parser.add_argument("--learning_rate",
                        help="learning rate", default=0.001, type=float)
    parser.add_argument(
        "--optimizer", help="optimizer algorithm", default='sgd', type=str)
    parser.add_argument(
        "--epochs", help="number of training epochs", default=100, type=int)
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
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            hp = yaml.safe_load(f)
    else:
        hp = {}

    hyperparams = {"model": hp["model"] if "model" in hp else args.model,
                   "num_sensors": hp["num_sensors"] if "num_sensors" in hp else args.num_sensors,
                   "dataset":  hp["dataset"] if "dataset" in hp else args.dataset,
                   "epochs": hp["epochs"] if "epochs" in hp else args.epochs,
                   "batch_size": hp["batch_size"] if "batch_size" in hp else args.batch_size,
                   "seq_len": hp["seq_len"] if "seq_len" in hp else args.seq_len,
                   "learning_rate": hp["learning_rate"] if "learning_rate" in hp else args.learning_rate,
                   "optimizer": hp["optimizer"] if "optimizer" in hp else args.optimizer,
                   "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')}
    if hyperparams["model"] == "transformer":
        hyperparams.update(
            {"d_model": hp["d_model"] if "d_model" in hp else args.d_model,
             "embedding_size_src": hp["embedding_size_src"] if "embedding_size_src" in hp else args.embedding_size_src,
             "embedding_size_tgt": hp["embedding_size_tgt"] if "embedding_size_tgt" in hp else args.embedding_size_tgt,
             "num_heads": hp["num_heads"] if "num_heads" in hp else args.n_heads,
             "dim_feedforward": hp["dim_feedforward"] if "dim_feedforward" in hp else args.dim_feedforward,
             "dropout": hp["dropout"] if "dropout" in hp else args.dropout,
             "num_encoder_layers": hp["num_encoder_layers"] if "num_encoder_layers" in hp else args.num_encoder_layers,
             }
        )

    return dict(hyperparams)


def load_model(hyperparams):
    """ Creates models based on hyperparameters.

    Args:
        hyperparams (dict): dict containing hyperparameters

    Returns:
        model (torch.nn.Module): Model based on hyperparameters
    """

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if hyperparams["model"] == "lstm":
        model = CustomLSTM(input_size=hyperparams["num_sensors"], hidden_size=hyperparams["num_sensors"]).to(
            device=device, non_blocking=True)
    elif hyperparams["model"] == "transformer":
        model = TransformerEncoder(**hyperparams).to(device=device, non_blocking=True)
    else:
        model = None

    return model


def load_optimizer(model, hyperparams):
    """Returns optimizer based on hyperparameters

    Args:
        model (torch.nn.Module): Torch model
        hyperparams (dict): Dict containing hyperparameters

    Returns:
        optimizer (torch.optim): Optimizer
    """

    if hyperparams["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyperparams["learning_rate"])
    elif hyperparams["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=hyperparams["learning_rate"])
    else:
        optimizer = None

    return optimizer
