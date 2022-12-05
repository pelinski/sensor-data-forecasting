import wandb
import torch
import yaml
import argparse
import pprint as pp

from models.lstm import CustomLSTM
from models.transformer import TransformerEncoder


def load_hyperparams():

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
                   "optimizer": hp["optimizer"] if "optimizer" in hp else "adam"}

    if hp["model"] == "transformer" and "transformer" in hp:
        hyperparams.update({"transformer_params":
                            {"d_model": hp["transformer"]["d_model"] if "d_model" in hp["transformer"] else 64,
                             "embedding_size_src": hp["transformer"]["embedding_size_src"] if "embedding_size_src" in hp["transformer"] else 8,
                             "embedding_size_tgt": hp["transformer"]["embedding_size_tgt"] if "embedding_size_tgt" in hp["transformer"] else 8,
                             "num_heads": hp["transformer"]["num_heads"] if "num_heads" in hp["transformer"] else 16,
                             "dim_feedforward": hp["transformer"]["dim_feedforward"] if "dim_feedforward" in hp["transformer"] else 256,
                             "dropout": hp["transformer"]["dropout"] if "dropout" in hp["transformer"] else 0.2,
                             "num_encoder_layers": hp["transformer"]["num_encoder_layers"] if "num_encoder_layers" in hp["transformer"] else 7,
                             }
                            })

    return dict(hyperparams)


def load_model(hyperparams):

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if hyperparams["model"] == "lstm":
        model = CustomLSTM(input_size=hyperparams["num_sensors"], hidden_size=hyperparams["num_sensors"]).to(
            device=device, non_blocking=True)
    elif hyperparams["model"] == "transformer":
        model = TransformerEncoder(
            **hyperparams["transformer_params"]).to(device=device, non_blocking=True)
    else:
        model = None

    return model


def load_optimizer(model, hyperparams):

    if hyperparams["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyperparams["learning_rate"])
    elif hyperparams["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=hyperparams["learning_rate"])
    else:
        optimizer = None

    return optimizer
