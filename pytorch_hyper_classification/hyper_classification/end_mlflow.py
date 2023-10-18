import mlflow

# Check for and end any active runs
if mlflow.active_run():
    mlflow.end_run()
    print("active runs ended")

mlflow.end_run()
