import wandb

run = wandb.init(project="proto")
wandb.run.log_code(".")

for i in range(15):
    run.log({"i": i})