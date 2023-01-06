import torch
import wandb
import pickle


def save_model(model, optimizer, scheduler, hyperparams, epoch):
    """Model saver method

    Args:
        model (torch.nn.module): torch model
        optimizer (torch.optim): optimizer object
        scheduler (torch.optim): scheduler object
        hyperparams (dict): dict with hyperparameters
        epoch (int): current epoch
    """

    save_filename = "{}/run_{}_epoch_{}.model".format(
        wandb.run.dir, wandb.run.id, epoch)

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), 'hyperparams': hyperparams}, save_filename)

    wandb.save("{}".format(save_filename), base_path=wandb.run.dir)


def save_params(params, wandb_run):
    """Save hyperparameters to wandb

    Args:
        params (dict): _description_
        wandb_run (wandb.sdk.wandb_run.Run): wandb run object
    """
    save_filename = "{}/{}-params.pk".format(
        wandb_run.dir, wandb_run.id)
    
    with open(save_filename, 'wb') as f:
        pickle.dump(params, f)