import os
from random import random, randint

import mlflow
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    print("Running mlflow_tracking.py")

    for run in ['a', 'b', 'c']:
        with mlflow.start_run(run_name=run):
            log_param("param1", randint(0, 100))

            log_metric("foo", random())
            log_metric("foo", random() + 1)
            log_metric("foo", random() + 2)

            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            with open("outputs/test.txt", "w") as f:
                f.write("hello world!")

            log_artifacts("outputs")
