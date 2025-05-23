# pre-check before inference
import copy
_monitor_model = None

def monitor(model):
    global _monitor_model
    _monitor_model = copy.deepcopy(model)

def inference(x, model):
    if model != _monitor_model:
        del model
        return _monitor_model(x)
    return model(x)
