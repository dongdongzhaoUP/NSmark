import logging

from .mlm import MLMVictim
from .plm import PLMVictim
from .Qmlm import QMLMVictim
from .Qplm import QPLMVictim
from .Qsc import QSCVictim
from .sc import SCVictim

VICTIM_LIST = {
    "plm": PLMVictim,
    "Qplm": QPLMVictim,
    "mlm": MLMVictim,
    "Qmlm": QMLMVictim,
    "sc": SCVictim,
    "Qsc": QSCVictim,
}


def get_victim(config):
    logging.info("\n> Loading {} from {} <\n".format(config.type, config.path))
    victim = VICTIM_LIST[config.type](config)
    return victim
