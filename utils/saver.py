import torch
import wandb


def save_model(model, optimizer, hyperparams, epoch):

    save_filename = "{}/run_{}_epoch_{}.model".format(
        wandb.run.dir, wandb.run.id, hyperparams["epochs"])

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'hyperparams': hyperparams}, save_filename)

    wandb.save("{}/models/".format(save_filename), base_path=wandb.run.dir)
