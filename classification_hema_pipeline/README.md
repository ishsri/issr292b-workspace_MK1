# classification_hema
training pipeline for classification in hematology project

# setup
run setup_env.sh

# run inference
python3 inference.py --model_path /path/to/model.pth --json_path /path/to/json.json --batch-size 64 --workers 4

# run inference without multithreading
python3 inference.py --model_path /path/to/model.pth --json_path /path/to/json.json --single_thread --batch-size 64

# run inference without batched input, also turns off multithreading
python3 inference.py --model_path /path/to/model.pth --json_path /path/to/json.json --not_batched