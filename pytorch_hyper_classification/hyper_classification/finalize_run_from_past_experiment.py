# %%
import mlflow    
import optuna
from optuna.samplers import TPESampler
from hyper_classification.hyper_wsi import start_and_track 
import argparse
from importlib import reload
from mlflow.tracking.client import MlflowClient

from hyper_classification.hyper_wsi import check_memory_wrapper, required_memory_approximation

# %%
parser = argparse.ArgumentParser(description='PyTorch Classification Training')
  
parser.add_argument('--orig-experiment-best-trial-id', dest="orig_experiment_best_trial_id", required=True, help='optuna trial_id')
parser.add_argument('--orig-experiment-id', dest="orig_experiment_id", required=True, help='mlflow experiment id')
parser.add_argument('--target-experiment-id', dest="target_experiment_id", required=True, help='mlflow experiment id')
parser.add_argument('-b', '--batch-size', default=1, type=int)
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-cv', '--cross-validation', dest="nr_cv", default=5, type=int)
args = parser.parse_args()

# %%
orig_experiment_best_trial_id = args.orig_experiment_best_trial_id # 202
orig_experiment_id = args.orig_experiment_id # 44
target_experiment_id = args.target_experiment_id # 57
batch_size = args.batch_size
epochs = args.epochs
nr_cv = args.nr_cv

# %%
# Via mlflow
mlflow.set_tracking_uri("http://mlflow.172.26.62.216.nip.io/")
client = MlflowClient()
orig_experiment = client.get_experiment(orig_experiment_id)
runs = mlflow.search_runs(experiment_ids=[orig_experiment_id])
orig_best_run = runs[runs["tags.mlflow.runName"] == str(orig_experiment_best_trial_id)].iloc[0]

target_experiment = client.get_experiment(target_experiment_id)
runs = mlflow.search_runs(experiment_ids=[target_experiment_id])
example_target_run = runs.iloc[-1]
target_experiment_basename = target_experiment.name[0:-len("_pytorch01")] 

# %% Let's not use optuna for finalizing, we do not need to prune and our parameters are fixed
# so mlflow is enough. Additionally mlflow contains all "extra" args from command line 
# and "None" for parameters that got introduced later (or in this case in new target experiment). 
params = [key for key in example_target_run.keys() if key[0:7] == "params."]
# Let's drop old params
orig_best_run = orig_best_run.drop([key for key in orig_best_run.keys() if key not in params])
for param in params:
    if param not in orig_best_run or orig_best_run[param] is None:
        print(f"{param} unset, you might need to set it manually! Copying example run value: {example_target_run[param]}")
        orig_best_run[param] = example_target_run[param]
    # Let's eval all strings to get back correct types
    try:
        orig_best_run[param] = eval(orig_best_run[param])
    except:
        pass
    
# %% Manually adjustment!
orig_best_run["params.experiment_name"] = target_experiment.name
orig_best_run["params.epochs"] = epochs
orig_best_run["params.batch_size"] = batch_size
# For cv runs set all other paths to None
orig_best_run["params.data_paths"] = None
orig_best_run["params.output_dir"] = None
orig_best_run["params.imgs_path"] = "/"
orig_best_run["params.balance_samples"] = True
orig_best_run["params.oversample"] = False
orig_best_run["params.log_model"] = False # Models are saved locally anyway and paths are registered in output_dir, so no need to waste more disk space
orig_best_run["params.log_roc"] = False  # This just fails too often, better to execute all visualizations after training
orig_best_run["params.name"] = f'experiment{orig_experiment_id}_run{orig_best_run["params.name"]}_finalized'

# %% convert all params to trial_args and all tags to user_attributes
# mlflow doesn't have a distinction between user_attributes and system_attributes,
# but we won't use optuna anyway, so converting all tags to user_attributes is fine
# or not... mlflow sets its own tags, which we shouldn't copy (e.g. tags.mlflow.user)
trial_args = argparse.Namespace()
trial_args.__dict__ = {key[7:]:orig_best_run[key] for key in params}
user_attr = {"finalize": True}

# %% Going for cross-validation we need to set CV-specific parameters inside loop
trial_args.nr_cv = nr_cv
trial_args.output_dir = f"/home/s7740678/workspaces/beegfs/s7740678-data_02/mds_diagnosis/tasks/results/{target_experiment_basename}/pytorch_01/"+"cv{cv}"+f"/{trial_args.name}"
trial_args.data_paths = [f"/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/{target_experiment_basename}/base_config/trainval/"+"cv{cv}"+f"/train.csv", f"/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/{target_experiment_basename}/base_config/trainval/"+"cv{cv}"+f"/val.csv"]
# trial_args.cv_output_dirs = []
# trial_args.cv_data_paths = []
# for i in range(trial_args.nr_cv):
#     trial_args.cv_output_dirs.append(f"/home/s7740678/workspaces/beegfs/s7740678-data_02/mds_diagnosis/tasks/results/{target_experiment_basename}/pytorch_01/cv{i}/{trial_args.name}")
#     trial_args.cv_data_paths.append([f"/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/{target_experiment_basename}/base_config/trainval/cv{i}/train.csv", f"/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/{target_experiment_basename}/base_config/trainval/cv{i}/val.csv"])

# %%
min_memory = required_memory_approximation(trial_args)
check_memory = check_memory_wrapper(min_memory)
check_memory(None, None)
start_and_track(None, trial_args, user_attr)

# %%
# output_dir /home/s7740678/workspaces/beegfs/s7740678-data_02/mds_diagnosis/tasks/results/aml_healthy_e2e_classification/pytorch_01/cv0/62
# data_paths	['/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/aml_healthy_e2e_classification/base_config/trainval/cv0/train.csv', '/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/aml_healthy_e2e_classification/base_config/trainval/cv0/val.csv']


# fixed_best_trial = DEBUG_TRIAL
# fixed_best_trial.number = 0 if not tracking_helper.optkeras.study.trials else tracking_helper.optkeras.study.trials[-1].number + 1
# fixed_best_trial.params = best_params

# # TODO get augmentation hash from trial
# # augmentation_name = "1613827913427995017"
# # transformed_objects_dir=Path("/beegfs/global0/ws/s7740678-apl_classification_tmp/transformed/apl_vs_healthy_wsi_classification_versioned/wsi/")
# mlflow.set_tracking_uri("http://mlflow.172.26.62.216.nip.io/")
# mlflow.set_experiment(config.task_name)
# mlflow.start_run(run_name=f"{best_trial_id}_finetuning")
# %%
