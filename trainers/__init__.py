from .finetune_trainer_wbw import FineTuneWBWTrainer
from .wbw_trainer import WBWTrainer
from .wbw_verifier import WBWVerifier

TRAINERS_LIST = {
    "wbw": WBWTrainer,
    "finetune_wbw": FineTuneWBWTrainer,
}


def get_trainer(config, save_dir):
    trainer = TRAINERS_LIST[config.method](config, save_dir)
    return trainer


def get_verifier(
    config, save_dir, model, proj, verify_dataset, sig, sm, sig_sm, poisoner, insert_num
):
    verifier = WBWVerifier(
        config,
        save_dir,
        model,
        proj,
        verify_dataset,
        sig,
        sm,
        sig_sm,
        poisoner,
        insert_num,
    )
    return verifier
