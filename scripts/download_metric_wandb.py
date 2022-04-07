import wandb
api = wandb.Api()

# %%
# run is specified by <entity>/<project>/<run_id>
run = api.run("lucyzmf/DHPC_morph_test_6/f5kupjdq")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()

# %%
metrics_dataframe.columns

# %%
metrics_dataframe.to_csv("/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs/wandb_metric_log/metrics_reset_per_frame_true.csv")
