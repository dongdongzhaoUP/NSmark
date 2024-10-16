import logging

from .plain_text_dataset import Plain_Text_Dataset
from .single_text_dataset import Single_Text_Dataset

DATA_ATTR = {
    "sst2": {"path": "data/sst2", "type": "single"},
    "offenseval": {"path": "data/offenseval", "type": "single"},
    "lingspam": {"path": "data/lingspam", "type": "single"},
    "sst5": {"path": "data/sst5", "type": "single"},
    "agnews": {"path": "data/agnews", "type": "single"},
    "wikitext-2": {"path": "data/wikitext-2", "type": "plain"},
}


DATASETS = {
    "single": Single_Text_Dataset,
    "plain": Plain_Text_Dataset,
}


def get_dataset(task):
    data_type = DATA_ATTR[task]["type"]
    Dataset = DATASETS[data_type](DATA_ATTR[task]["path"])
    if data_type == "tc":
        dataset, num_labels, real_labels = Dataset()
    else:
        dataset = Dataset()

    logging.info("\n========= Load dataset ==========")
    logging.info("{} Dataset : ".format(task))
    if data_type != "qa":
        logging.info(
            "\tTrain : {}\n\tDev : {}\n\tTest : {}".format(
                len(dataset["train"]), len(dataset["dev"]), len(dataset["test"])
            )
        )
    logging.info("-----------------------------------")

    if data_type == "tc":
        return dataset, num_labels, real_labels
    else:
        return dataset
