#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --partition=ml
#SBATCH --output=/beegfs/ws/1/issr292b-workspace_MK1/issr292b-workspace_MK1/logs/e2e_hpt_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16  # number of processor cores (i.e. threads)
#SBATCH --mem=32G
#SBATCH -J "e2e_classification" # job-name
#SBATCH --mail-user=tim.schmittmann@mailbox.tu-dresden.de   # email address
#SBATCH --mail-type=FAIL
# alpha:
# --partition=alpha
# --array 0-11,20-31,40-51,60-71,80-91,200-211,220-231,240-251,260-271,280-291
# ml
# --partition=ml
# --array 100-111,120-131,140-151,160-171,180-191,300-311,320-331,340-351,360-371,380-391

if [ ! -z $1 ]; then
  SLURM_ARRAY_TASK_ID=$1
fi

PARTITION=${HOSTNAME:6:2}
echo "Hi, I am step $SLURM_ARRAY_TASK_ID in this array job $SLURM_ARRAY_JOB_ID on $PARTITION"

if [ "$PARTITION" = "ml" ]; then
  echo "Activate ml_pipeline_ppc64le ($(which conda))"
  eval "$(conda shell.bash hook)"
  conda activate ml_pipeline_ppc64le
  module load modenv/ml  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5
  echo "Load modenv/ml  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5"
else
  echo "Activate ml_pipeline_x86_64 ($(which conda))"
  eval "$(conda shell.bash hook)"
  conda activate ml_pipeline_x86_64
  module load modenv/hiera  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5
  echo "Load modenv/hiera  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5"
fi
echo "Check Python: $(which python)"
python -c 'import torch; import torchvision; print(f"Torch v{torch.__version__}, Torchvision v{torchvision.__version__}"); print(f"Cuda available: {torch.cuda.is_available()}")'
echo "Run script"

EXPERIMENT_SCRIPT=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/libs/bms_ml_tools/bms_ml_tools/pytorch_pipeline/libs/hyper_classification/hyper_classification/create_experiment_study.py
SCRIPT=/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/libs/bms_ml_tools/bms_ml_tools/pytorch_pipeline/libs/hyper_classification/hyper_classification/hyper_wsi.py
HEIGHT=1920
WIDTH=2560
EPOCHS=3
BATCH_SIZE=1
WORKERS=4
LOG_ROC="--no-log-roc"
LOG_MODEL="--no-log-model"
export PYTHONHASHSEED=0
export DISABLE_GPU=False
export TORCH_HOME=/home/s7740678/workspaces/beegfs/s7740678-condahome_02/x86_64/.torch
NAME=NONOAME
CLASSNAME_0=ZERO
CLASSNAME_1=ONE
TOTAL_TASKS=1
if [ $(( $SLURM_ARRAY_TASK_ID % $TOTAL_TASKS )) -eq 0 ]
then
  NAME=mds_eb_vs_d_e2e_bg_demo
  CLASSNAME_0=eb
  CLASSNAME_1=d
fi
TRAIN_PATH="/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv{cv}/train.csv"
TEST_PATH="/home/s7740678/p_ai4hematology/code/s7740678/mds_diagnosis/tasks/configs/$NAME/base_config/trainval/cv{cv}/val.csv"
OUTPUT_DIR="/home/s7740678/workspaces/beegfs/s7740678-data_02/mds_diagnosis/tasks/results/$NAME/pytorch_01/cv{cv}/"
EXPERIMENT_NAME="${NAME}_pytorch01"
MIN_MEMORY=33300000000 #  --min-memory $MIN_MEMORY 

echo "$NAME cv0"
echo "python $SCRIPT -j $WORKERS $LOG_ROC $LOG_MODEL -b $BATCH_SIZE --epochs $EPOCHS --input-size $HEIGHT $WIDTH --class-names $CLASSNAME_0 $CLASSNAME_1 --data-paths $TRAIN_PATH $TEST_PATH --experiment-name $EXPERIMENT_NAME --output-dir $OUTPUT_DIR"
# python $EXPERIMENT_SCRIPT --name $EXPERIMENT_NAME --db 'postgresql://mlflow_user:mlflow@172.26.62.216:6543/optuna'
python $SCRIPT -j $WORKERS $LOG_ROC $LOG_MODEL -b $BATCH_SIZE --epochs $EPOCHS --input-size $HEIGHT $WIDTH --class-names $CLASSNAME_0 $CLASSNAME_1 --data-paths $TRAIN_PATH $TEST_PATH --experiment-name $EXPERIMENT_NAME --output-dir $OUTPUT_DIR


