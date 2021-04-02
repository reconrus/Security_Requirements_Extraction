import numpy as np
from transformers import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            print(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping is disabled"
            )
            return

        if metric_value == 1:
            control.should_training_stop = True

