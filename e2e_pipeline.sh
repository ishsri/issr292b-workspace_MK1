#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --partition=ml
#SBATCH --output=/beegfs/ws/1/issr292b-workspace_MK1/logs/e2e_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16  # number of processor cores (i.e. threads)
#SBATCH --mem=32G
#SBATCH -J "e2e_classifier" # job-name
#SBATCH --mail-user=ishan.srivastava@tu-dresden.de   # email address
#SBATCH --mail-type=FAIL

PARTITION=${HOSTNAME:6:2}

NAME=NONAME
export PATH="/beegfs/ws/1/issr292b-workspace_MK1/pytorch_hyper_classification/hyper_classification/:$PATH"


echo "Hi, I am running e2e pipeline on $PARTITION"

if [ "$PARTITION" = "ml" ]; then
  echo "Activate final env ($(which conda))"
  eval "$(conda shell.bash hook)"
  conda activate /beegfs/ws/1/issr292b-workspace_MK1/envs/final/
  module load modenv/ml  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5
  echo "Load modenv/ml  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5"
  
else
  echo "Unsupported partition: $PARTITION"
  exit 1  # Exit with an error code

fi

echo "Check Python: $(which python)"
python -c 'import torch; import torchvision; print(f"Torch v{torch.__version__}, Torchvision v{torchvision.__version__}"); print(f"Cuda available: {torch.cuda.is_available()}")'
echo "Run script"

echo "$NAME cv0"


