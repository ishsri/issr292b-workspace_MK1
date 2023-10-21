#!/bin/bash -l
#SBATCH --array 1-4,6-9,11-14,16-19
#SBATCH --gres=gpu:1
#SBATCH --partition=alpha
#SBATCH --output=/beegfs/ws/1/issr292b-workspace_MK1/issr292b-workspace_MK1/logs/e2e_hpt_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16  # number of processor cores (i.e. threads)
#SBATCH --mem=16G
#SBATCH -J "e2e_hyperparamtuning_classification" # job-name
#SBATCH --mail-user=ishan.srivastava@tu-dresden.de   # email address
#SBATCH --mail-type=FAIL

SLURM_ARRAY_TASK_ID=$1

echo "Hi, I am step $SLURM_ARRAY_TASK_ID in this array job $SLURM_ARRAY_JOB_ID"
eval "$(conda shell.bash hook)"
echo "Activate ml_pipeline_x86_64"
conda activate ml_pipeline_x86_64
echo "Load modenv/hiera  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5"
module load modenv/hiera  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5
echo "Check Python: $(which python)"
python -c 'import torch; import torchvision; print(f"Torch v{torch.__version__}, Torchvision v{torchvision.__version__}"); print(f"Cuda available: {torch.cuda.is_available()}")'
echo "Run script"

EXPERIMENT_SCRIPT=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/libs/bms_ml_tools/bms_ml_tools/pytorch_pipeline/libs/hyper_classification/hyper_classification/create_experiment_study.py
SCRIPT=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/libs/bms_ml_tools/bms_ml_tools/pytorch_pipeline/libs/hyper_classification/hyper_classification/hyper_wsi.py
HEIGHT=1920
WIDTH=2560
EPOCHS=30
BATCH_SIZE=2
WORKERS=4
LOG_ROC="--no-log-roc"
LOG_MODEL="--no-log-model"
export PYTHONHASHSEED=0
export DISABLE_GPU=False
export TORCH_HOME=/home/s7740678/workspaces/beegfs/s7740678-condahome_02/x86_64/.torch

if [ $(( $SLURM_ARRAY_TASK_ID % 5 )) -eq 1 ]
then
  NAME=aml_vs_healthy_e2e_bg
  TRAIN_PATH=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv0/train.csv
  TEST_PATH=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv0/val.csv
  OUTPUT_DIR=/home/s7740678/workspaces/beegfs/s7740678-data_02/mds_diagnosis/tasks/results/$NAME/pytorch_01/cv0/
  EXPERIMENT_NAME="${NAME}_pytorch01"
  CLASSNAME_0=aml
  CLASSNAME_1=healthy
elif [ $(( $SLURM_ARRAY_TASK_ID % 5 )) -eq 2 ]
then
  NAME=mds_vs_healthy_e2e_bg
  # echo "$NAME cv0"
  TRAIN_PATH=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv0/train.csv
  TEST_PATH=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv0/val.csv
  OUTPUT_DIR=/home/s7740678/workspaces/beegfs/s7740678-data_02/mds_diagnosis/tasks/results/$NAME/pytorch_01/cv0/
  EXPERIMENT_NAME="${NAME}_pytorch01"
  CLASSNAME_0=mds
  CLASSNAME_1=healthy
  # python $EXPERIMENT_SCRIPT --name $EXPERIMENT_NAME --db 'postgresql://mlflow_user:mlflow@172.26.62.216:6543/optuna'
  # python $SCRIPT -j $WORKERS $LOG_ROC $LOG_MODEL -b $BATCH_SIZE --epochs $EPOCHS --input-size $HEIGHT $WIDTH  --class-names $CLASSNAME_0 $CLASSNAME_1 --data-paths $TRAIN_PATH $TEST_PATH --experiment-name $EXPERIMENT_NAME --output-dir $OUTPUT_DIR
elif [ $(( $SLURM_ARRAY_TASK_ID % 5 )) -eq 3 ]
then
  NAME=mds_eb_vs_d_e2e_bg
  # echo "$NAME cv0"
  TRAIN_PATH=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv0/train.csv
  TEST_PATH=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv0/val.csv
  OUTPUT_DIR=/home/s7740678/workspaces/beegfs/s7740678-data_02/mds_diagnosis/tasks/results/$NAME/pytorch_01/cv0/
  EXPERIMENT_NAME="${NAME}_pytorch01"
  CLASSNAME_0=mds_d
  CLASSNAME_1=mds_eb
  # python $EXPERIMENT_SCRIPT --name $EXPERIMENT_NAME --db 'postgresql://mlflow_user:mlflow@172.26.62.216:6543/optuna'
  # python $SCRIPT -j $WORKERS $LOG_ROC $LOG_MODEL -b $BATCH_SIZE --epochs $EPOCHS --input-size $HEIGHT $WIDTH  --class-names $CLASSNAME_0 $CLASSNAME_1 --data-paths $TRAIN_PATH $TEST_PATH --experiment-name $EXPERIMENT_NAME --output-dir $OUTPUT_DIR
elif [ $(( $SLURM_ARRAY_TASK_ID % 5 )) -eq 4 ]
then
  NAME=mds_rs_vs_non_rs_e2e_bg
  # echo "$NAME cv0"
  TRAIN_PATH=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv0/train.csv
  TEST_PATH=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv0/val.csv
  OUTPUT_DIR=/home/s7740678/workspaces/beegfs/s7740678-data_02/mds_diagnosis/tasks/results/$NAME/pytorch_01/cv0/
  EXPERIMENT_NAME="${NAME}_pytorch01"
  CLASSNAME_0=rs
  CLASSNAME_1=non_rs
  # python $EXPERIMENT_SCRIPT --name $EXPERIMENT_NAME --db 'postgresql://mlflow_user:mlflow@172.26.62.216:6543/optuna'
  # python $SCRIPT -j $WORKERS $LOG_ROC $LOG_MODEL -b $BATCH_SIZE --epochs $EPOCHS --input-size $HEIGHT $WIDTH  --class-names $CLASSNAME_0 $CLASSNAME_1 --data-paths $TRAIN_PATH $TEST_PATH --experiment-name $EXPERIMENT_NAME --output-dir $OUTPUT_DIR
fi
TRAIN_PATH=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv0/train.csv
TEST_PATH=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv0/val.csv
OUTPUT_DIR=/home/s7740678/workspaces/beegfs/s7740678-data_02/mds_diagnosis/tasks/results/$NAME/pytorch_01/cv0/
EXPERIMENT_NAME="${NAME}_pytorch01"

echo "$NAME cv0"
echo "python $SCRIPT -j $WORKERS $LOG_ROC $LOG_MODEL -b $BATCH_SIZE --epochs $EPOCHS --input-size $HEIGHT $WIDTH --class-names $CLASSNAME_0 $CLASSNAME_1 --data-paths $TRAIN_PATH $TEST_PATH --experiment-name $EXPERIMENT_NAME --output-dir $OUTPUT_DIR"
# python $EXPERIMENT_SCRIPT --name $EXPERIMENT_NAME --db 'postgresql://mlflow_user:mlflow@172.26.62.216:6543/optuna'
python $SCRIPT -j $WORKERS $LOG_ROC $LOG_MODEL -b $BATCH_SIZE --epochs $EPOCHS --input-size $HEIGHT $WIDTH --class-names $CLASSNAME_0 $CLASSNAME_1 --data-paths $TRAIN_PATH $TEST_PATH --experiment-name $EXPERIMENT_NAME --output-dir $OUTPUT_DIR
