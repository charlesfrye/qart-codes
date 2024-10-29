import wandb

run = wandb.init(project="proto")

for i in range(10):
    run.log({"i": i})