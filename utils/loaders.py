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
        "-c", "--config",
        help="yaml config file",
        default="configs/test.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        hp = yaml.safe_load(f)

    # Hyperparameters either come from wandb or are default values
    hyperparams = {"model": hp["model"] if "model" in hp else "lstm",
                   "num_sensors": hp["num_sensors"] if "num_sensors" in hp else 2,
                   "data_path":  hp["data_path"] if "data_path" in hp else "dataset/data/chaos-bells-2/processed/RX0",
                   "epochs": hp["epochs"] if "epochs" in hp else 5,
                   "batch_size": hp["batch_size"] if "batch_size" in hp else 32,
                   "sequence_length": hp["sequence_length"] if "sequence_length" in hp else 16,
                   "learning_rate": hp["learning_rate"] if "learning_rate" in hp else 0.001,
                   "optimizer": hp["optimizer"] if "optimizer" in hp else "adam",
                   "model_params": hp["model_params"] if "model_params" in hp else {},
                   "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')}
    if hp["model"] == "transformer":
        hyperparams.update({"model_params":
                            {"d_model": hyperparams["model_params"]["d_model"] if "d_model" in hyperparams["model_params"] else 64,
                             "embedding_size_src": hyperparams["model_params"]["embedding_size_src"] if "embedding_size_src" in hyperparams["model_params"] else 8,
                             "embedding_size_tgt": hyperparams["model_params"]["embedding_size_tgt"] if "embedding_size_tgt" in hyperparams["model_params"] else 8,
                             "num_heads": hyperparams["model_params"]["num_heads"] if "num_heads" in hyperparams["model_params"] else 16,
                             "dim_feedforward": hyperparams["model_params"]["dim_feedforward"] if "dim_feedforward" in hyperparams["model_params"] else 256,
                             "dropout": hyperparams["model_params"]["dropout"] if "dropout" in hyperparams["model_params"] else 0.2,
                             "num_encoder_layers": hyperparams["model_params"]["num_encoder_layers"] if "num_encoder_layers" in hyperparams["model_params"] else 7,
                             }
                            })

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
        model = TransformerEncoder(
            **hyperparams, **hyperparams["model_params"]).to(device=device, non_blocking=True)
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
