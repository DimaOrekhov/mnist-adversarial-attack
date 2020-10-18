from catalyst.core import Callback, CallbackOrder
from typing import List, Union, Optional, Dict


class SaveAllBatchMetricsCallback(Callback):

    def __init__(self, logging_dict: Dict, metric_keys: Optional[Union[str, List[str]]] = None):
        super().__init__(order=CallbackOrder.Logging)
        self.logging_dict = logging_dict
        self.metric_keys = metric_keys

    def on_batch_end(self, runner: "IRunner"):
        if self.metric_keys is None:
            # log everything
            return self.logging_dict[runner.loader_name].append(runner.batch_metrics)
        self.logging_dict[runner.loader_name].append(
            {key: runner.batch_metrics[key] for key in self.metric_keys}
        )
