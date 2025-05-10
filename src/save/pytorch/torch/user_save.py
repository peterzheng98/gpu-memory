# pre-check before inference
_monitor_model = None

def monitor(model):
    _monitor_model = model

def inference(x, model):
    if model != _monitor_model:
        return _monitor_model(x)
    return model(x)