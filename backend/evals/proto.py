import wandb

run = wandb.init(project="proto")

for i in range(15):
    run.log({"i": i})