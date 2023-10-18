conda create -y  -p /beegfs/ws/1/issr292b-workspace_MK1/envs/train_class python=3.7.10
#conda create -y -p /beegfs/ws/0/s2558947-bee_ws/envs/train_class python=3.7.10
conda activate /beegfs/ws/1/issr292b-workspace_MK1/envs/train_class
#conda install -y pytorch=1.12.1 torchvision=0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -y pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia

conda install -y pandas=1.3.5
conda install -y torchmetrics=0.10.0 -c conda-forge
#conda install -c conda-forge mlflow
#conda install -y -c conda-forge pillow=9.2.0
#conda install -c conda-forge pycocotools

conda install -y ipykernel
python -m ipykernel install --user --name class_3_7_10 --display-name="class_3_7_10"

conda install mlflow=1.29.0 scikit-learn=1.0.2 optuna=2.10.1 pyyaml=6.0 setuptools=65.6.3 matplotlib=22.9.0

_openmp_mutex=5.1 