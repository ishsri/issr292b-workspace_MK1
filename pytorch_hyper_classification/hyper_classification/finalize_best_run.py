# %%
import mlflow    
import optuna
from optuna.samplers import TPESampler
from hyper_classification.hyper_wsi import start_and_track 
import argparse
from importlib import reload
from mlflow.tracking.client import MlflowClient

from hyper_classification.hyper_wsi import check_memory_wrapper, required_memory_approximation
import numpy as np
# %% 
# best_trial_id = 60
# experiment_id = 55
# batch_size = 1
# epochs = 60
# nr_cv = 5
# cv_start = 1

# %%
parser = argparse.ArgumentParser(description='PyTorch Classification Training')

parser.add_argument('--run-to-resume-uuid', dest="run_to_resume_uuid", help='')
parser.add_argument('--best-trial-id', dest="best_trial_id", help='optuna trial_id')
parser.add_argument('--experiment-id', dest="experiment_id", required=True, help='mlflow experiment id')
parser.add_argument('-b', '--batch-size', default=1, type=int)
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cv-start', dest="cv_start", default=0, type=int)
parser.add_argument('-cv', '--cross-validation', dest="nr_cv", default=5, type=int)
args = parser.parse_args()

# %%
run_to_resume_uuid = args.run_to_resume_uuid
best_trial_id = args.best_trial_id
experiment_id = args.experiment_id
batch_size = args.batch_size
epochs = args.epochs
nr_cv = args.nr_cv
cv_start = args.cv_start

# %%
# Via mlflow
mlflow.set_tracking_uri("http://mlflow.172.26.62.216.nip.io/")
client = MlflowClient()
experiment = client.get_experiment(experiment_id)
runs = mlflow.search_runs(experiment_ids=[experiment_id])
best_run_name = str(best_trial_id)
best_run = runs[runs["tags.mlflow.runName"] == best_run_name].iloc[0]
finalizing_run_to_resume = None
if run_to_resume_uuid:
    finalizing_run_to_resume = runs[runs["run_id"] == run_to_resume_uuid].iloc[0]
    
experiment_basename = experiment.name[0:-len("_pytorch01")] 

# Via optuna
study = optuna.load_study(study_name=experiment.name,
                            storage='postgresql://postgres:SvBU81olI4@172.26.62.216:6543/optuna')
best_trial = next(filter(lambda trial:  trial.number == int(best_trial_id), study.trials))

# print(run.data.params["data_paths"])
# #print('data: ', run.data)
# print('run obj: ', run.data.params)

#with mlflow.start_run() as run:


# %% Check best_run == best_trial and convert data types
# assert(best_run["params.name"] == str(best_trial_id))
#
# for param, value in best_trial.params.items():
#     print(param)
#     print(best_run[f"params.{param}"])
#     print(value)
#     if type(value) == float:
#         # Optuna rounds float values, all mlflow values are strings
#         print(round(float(best_run[f"params.{param}"]), 12))
#         assert(round(float(best_run[f"params.{param}"]), 12) == round(value, 12))
#     else:        
#         assert(best_run[f"params.{param}"] == str(value))
#     best_run[f"params.{param}"] = type(value)(best_run[f"params.{param}"])
# Faking run
# runs = mlflow.search_runs(experiment_ids=["77"])
# fake_run = runs[runs["run_id"] == "73c730ab84a84759a98896caaecc0822"].iloc[0]
# best_run = fake_run
# %% Let's not use optuna for finalizing, we do not need to prune and our parameters are fixed
# so mlflow is enough. Additionally mlflow contains all "extra" args from command line 
# and "None" for parameters that got introduced later

# param_keys = np.unique([
#     *best_run.keys(), "params.experiment_name", "params.epochs", "params.batch_size", "params.data_paths", "params.output_dir",
#     "params.imgs_path", "params.balance_samples", "params.balance_val_set", "params.oversample", "params.log_model",
#     "params.log_roc", "params.name"
# ])
params = [key for key in best_run.keys() if key[0:7] == "params."]
for param in params:
    if best_run[param] is None:
        print(f"{param} unset, you might need to set it manually!")
    # Let's eval all strings to get back correct types
    try:
        best_run[param] = eval(best_run[param])
    except:
        pass
    
# %% Manually adjustment!
best_run["params.experiment_name"] = experiment.name
best_run["params.epochs"] = epochs
best_run["params.batch_size"] = batch_size
# For cv runs set all other paths to None
best_run["params.data_paths"] = None
best_run["params.output_dir"] = None
best_run["params.imgs_path"] = "/"
best_run["params.balance_samples"] = True
best_run["params.balance_val_set"] = False # Maybe leave it at True?
best_run["params.oversample"] = False
best_run["params.log_model"] = False # Models are saved locally anyway and paths are registered in output_dir, so no need to waste more disk space
best_run["params.log_roc"] = False # This just fails too often, better to execute all visualizations arfter training
best_run["params.name"] = f'{best_run_name}_finalized'

# Repeat for manually set keys
params = [key for key in best_run.keys() if key[0:7] == "params."]

# %% convert all params to trial_args and all tags to user_attributes
# mlflow doesn't have a distinction between user_attributes and system_attributes,
# but we won't use optuna anyway, so converting all tags to user_attributes is fine
# or not... mlflow sets its own tags, which we shouldn't copy (e.g. tags.mlflow.user)
trial_args = argparse.Namespace()
trial_args.__dict__ = {key[7:]:best_run[key] for key in params}
user_attr = {"finalize": True}

# %% Going for cross-validation we need to set CV-specific parameters inside loop
trial_args.nr_cv = nr_cv
trial_args.output_dir = f"/home/s7740678/workspaces/beegfs/s7740678-data_02/mds_diagnosis/tasks/results/{experiment_basename}/pytorch_01/"+"cv{cv}"+f"/{trial_args.name}"
trial_args.data_paths = [f"/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/{experiment_basename}/base_config/trainval/"+"cv{cv}"+f"/train.csv", f"/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/{experiment_basename}/base_config/trainval/"+"cv{cv}"+f"/val.csv"]
# trial_args.cv_output_dirs = []
# trial_args.cv_data_paths = []
# for i in range(trial_args.nr_cv):
#     trial_args.cv_output_dirs.append(f"/home/s7740678/workspaces/beegfs/s7740678-data_02/mds_diagnosis/tasks/results/{experiment_basename}/pytorch_01/cv{i}/{trial_args.name}")
#     trial_args.cv_data_paths.append([f"/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/{experiment_basename}/base_config/trainval/cv{i}/train.csv", f"/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/{experiment_basename}/base_config/trainval/cv{i}/val.csv"])

# %%
print(trial_args)
run_uuid = None
if finalizing_run_to_resume is not None:
    print(f'Resuming Run {finalizing_run_to_resume["run_id"]}')
    run_uuid = finalizing_run_to_resume["run_id"]
min_memory = required_memory_approximation(trial_args)
check_memory = check_memory_wrapper(min_memory)
check_memory(None, None)
start_and_track(None, trial_args, user_attr, {}, cv_start, run_uuid)

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
