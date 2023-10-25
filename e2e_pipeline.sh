#!/bin/bash -l

#SBATCH --gres=gpu:1
#SBATCH --partition=alpha
#SBATCH --output=/beegfs/ws/1/issr292b-workspace_MK1/issr292b-workspace_MK1/logs/e2e_trial_%a.log
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
conda activate /beegfs/ws/1/issr292b-workspace_MK1/issr292b-workspace_MK1/envs/final
echo "Load modenv/hiera  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5"
module load modenv/hiera  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5
echo "Check Python: $(which python)"
python -c 'import torch; import torchvision; print(f"Torch v{torch.__version__}, Torchvision v{torchvision.__version__}"); print(f"Cuda available: {torch.cuda.is_available()}")'
echo "Run script"


SCRIPT=/beegfs/ws/1/issr292b-workspace_MK1/issr292b-workspace_MK1/pytorch_hyper_classification/hyper_classification/hyper_wsi.py

HEIGHT=1920
WIDTH=2560
EPOCH=30
BATCH_SIZE=2
WORKERS=4
LOG_ROC="--no-log-roc"
LOG_MODEL="--no-log-model"

NAME=mds_eb_vs_d_e2e_bg

TRAIN_PATH=/beegfs/ws/1/issr292b-workspace_MK1/model_files/mds_eb_vs_d_e2e_bg/base_config/trainval/cv{cv}/train.csv

TEST_PATH=/beegfs/ws/1/issr292b-workspace_MK1/model_files/mds_eb_vs_d_e2e_bg/base_config/trainval/cv{cv}/val.csv

OUTPUT_DIR=/beegfs/ws/1/issr292b-workspace_MK1/output_mds

EXPERIMENT_NAME="${NAME}_pytorch01"
CLASSNAME_0=d
CLASSNAME_1=mds_eb

echo "$NAME cv0"
echo "python $SCRIPT -j $WORKERS $LOG_ROC $LOG_MODEL -b $BATCH_SIZE --epochs $EPOCH --input-size $HEIGHT $WIDTH --class-names $CLASSNAME_0 $CLASSNAME_1 --data-paths $TRAIN_PATH $TEST_PATH --experiment-name $EXPERIMENT_NAME --output-dir $OUTPUT_DIR"

python $SCRIPT -j $WORKERS $LOG_ROC $LOG_MODEL -b $BATCH_SIZE --epochs $EPOCH --input-size $HEIGHT $WIDTH --class-names $CLASSNAME_0 $CLASSNAME_1 --data-paths $TRAIN_PATH $TEST_PATH --experiment-name $EXPERIMENT_NAME --output-dir $OUTPUT_DIR -cv 5 --cv-start 0