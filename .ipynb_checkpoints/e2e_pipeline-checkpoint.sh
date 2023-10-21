#!/bin/bash -l
#SBATCH --array 1-4
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
conda activate ml_pipeline_x86_64
echo "Load modenv/hiera  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5"
module load modenv/hiera  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5
echo "Check Python: $(which python)"
python -c 'import torch; import torchvision; print(f"Torch v{torch.__version__}, Torchvision v{torchvision.__version__}"); print(f"Cuda available: {torch.cuda.is_available()}")'
echo "Run script"

SCRIPT=