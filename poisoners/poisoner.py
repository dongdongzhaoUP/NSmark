import logging


class Poisoner(object):
    def __init__(self):
        pass

    def __call__(self):
        pass

    def poison_dataset(self, dataset):
        pass

    def poison_text(self, text):
        pass

    def show_dataset(self, dataset):
        logging.info("-----------------------------------")
        logging.info("Dataset info:")
        for key in dataset.keys():
            logging.info("\t{} : {}".format(key, len(dataset[key])))
        logging.info("-----------------------------------")

    def load_poison_data(self):
        pass

    def save_data(self):
        pass
