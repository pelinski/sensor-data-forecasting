import torch
import wandb


def save_model(model, optimizer, scheduler, hyperparams, epoch):
    
    save_filename = "{}/run_{}_epoch_{}.model".format(
        wandb.run.dir, wandb.run.id, epoch)

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), 'hyperparams': hyperparams}, save_filename)

    wandb.save("{}".format(save_filename), base_path=wandb.run.dir)
