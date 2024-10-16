import logging

import yaml


def get_config(path):
    return Config(path)


def load_yaml(filename):
    stream = open(filename, "r")
    docs = yaml.load_all(stream, Loader=yaml.FullLoader)
    config_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            config_dict[k] = v
    return config_dict


class Dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = Dotdict(value)
            self[key] = value


class Config(Dotdict):
    def __init__(self, path):
        super(Dotdict, self).__init__()
        config_dict = load_yaml(path)
        self.dotdict = Dotdict(config_dict)
        for k, v in self.dotdict.items():
            setattr(self, k, v)

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__

    def show_config(self):
        logging.info(f"\n============= Config ==============")
        for key, value in self.dotdict.items():
            if hasattr(value, "keys"):
                logging.info(f"{key} : ")
                for k, v in value.items():
                    logging.info(f"\t{k} : {v}")
            else:
                logging.info(f"{key} : {value}")
            logging.info(f"-----------------------------------")
