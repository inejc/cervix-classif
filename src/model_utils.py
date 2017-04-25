import json
import os
from time import gmtime, strftime

from keras.callbacks import Callback

from data_provider import FRCNN_MODELS_DIR


def dump_args(func):
    '''Decorator to print function call details - parameters names and effective values'''

    def wrapper(*func_args, **func_kwargs):
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        kwargs = {name: func_kwargs[name] for name in arg_names if name in func_kwargs.keys()}
        args = func_args[:len(arg_names)]
        params = {**dict(zip(arg_names, args)), **kwargs}
        defaults = func.__defaults__ or ()
        params = {**params, **{name: defaults[i - (len(arg_names) - len(defaults))] for i, name in enumerate(arg_names)
                               if name not in params.keys()}}
        model_dir = os.path.join(FRCNN_MODELS_DIR, params["model_name"])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open(os.path.join(model_dir, 'log.txt'), 'w') as log:
            log.write(strftime("%d.%m.%Y, %H:%M", gmtime()) + " --> ")
            log.write(func.__name__ + '(' + ', '.join('%s=%r' % (n, params[n]) for n in arg_names) + ')' + "\n")

        return func(*func_args, **func_kwargs)

    return wrapper


class LoggingCallback(Callback):
    """
    Callback that logs message at end of epoch.
    """

    def __init__(self, config):
        Callback.__init__(self)
        self.model_dir = os.path.join(FRCNN_MODELS_DIR, config.model_name)
        with open(os.path.join(self.model_dir, 'config.json'), 'w') as file:
            json.dump(config, file, default=lambda o: o.__dict__, indent=4, separators=(',', ': '))

    def on_epoch_end(self, epoch, logs={}):
        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        with open(os.path.join(self.model_dir, 'log.txt'), 'a') as log:
            log.write("\n" + msg)
