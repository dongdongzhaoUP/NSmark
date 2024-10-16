from .wbw_poisoner import WBWPoisoner

POISONERS_LIST = {"wbw": WBWPoisoner}


def get_poisoner(config):
    poisoner = POISONERS_LIST[config.method](config)
    print("> get poisoner:", config.method, "<")
    return poisoner
