import argparse
import logging
import os
from datetime import datetime
import numpy as np
import torch
from configs import get_config
from data import get_dataset
from decoder import Projection
from numpy import *
from numpy.linalg import *
from poisoners import get_poisoner
from trainers import get_trainer, get_verifier
from utils import *
from victims import get_victim

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="./configs/config.yaml")
args = parser.parse_args()
config = get_config(args.config_path)
load_and_print_config(args.config_path)

timestamp = int(datetime.now().timestamp())
config.model_save_dir = os.path.join(
    "./Models",
    f"{config.pretrain_trainer.method}",
    f"{config.victim.model}",
    f"{config.file_name}",
    str(timestamp),
)

config.result_save_dir_base = os.path.join(
    "./Results",
    f"{config.pretrain_trainer.method}",
    f"{config.victim.model}",
    f"{config.file_name}",
    str(timestamp),
)


print("\n> model_save_dir:", config.model_save_dir)
print("\n> result_save_dir_base:", config.result_save_dir_base)


set_logging(config.model_save_dir)
logging.info(config.model_save_dir)
logging.info(config.result_save_dir_base)

set_seed(config.seed)

device = torch.device("cuda")


def key_init(n=3):
    sig = 2 * np.random.randint(0, 2, size=(256)) - 1
    sig_repeat = torch.tensor(np.tile(sig, n)).to(device)
    sm = torch.tensor(2 * np.random.randint(0, 2, size=(n * 256)) - 1).to(device)
    sig_sm = (sig_repeat * sm).float().to(device)
    sig = torch.tensor(sig).to(device)
    proj = Projection().to(device)
    return sig_sm, proj, sig, sm


def load_key(key_path):
    sig_path = os.path.join(key_path, config.pretrain_trainer.sig_name)
    sm_path = os.path.join(key_path, config.pretrain_trainer.sm_name)
    sig_sm_path = os.path.join(key_path, config.pretrain_trainer.sig_sm_name)
    trigger_path = os.path.join(key_path, config.pretrain_trainer.trigger_name)
    proj_path = os.path.join(key_path, config.pretrain_trainer.proj_ckpt_name)

    proj = Projection()
    proj.load_state_dict(torch.load(proj_path, map_location=device))
    proj = proj.to(device)

    sig = torch.load(sig_path, map_location=device).to(device)
    sm = torch.load(sm_path, map_location=device).to(device)
    sig_sm = torch.load(sig_sm_path, map_location=device).to(device)

    if os.path.exists(trigger_path):
        f = open(trigger_path, "r")
        triggers = f.readlines()
        for i in range(len(triggers)):
            triggers[i] = triggers[i].strip()
        poisoner.set_triggers(triggers)

        return sig_sm, proj, sig, sm, triggers

    return sig_sm, proj, sig, sm


def prepare_model_for_saving(model):
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            print(f"Making {name} contiguous before saving")
            param.data = param.data.contiguous()
    return model


pretrain_dataset = get_dataset(config.dataset.pretrain)


if config.debug:
    for key in pretrain_dataset.keys():
        pretrain_dataset[key] = pretrain_dataset[key][:8]


plm_victim = get_victim(config.victim)


poisoner = get_poisoner(config.poisoner)
print("\n> Poisoner.triggers: ", poisoner.triggers)

if config.victim.do_train:
    logging.info("\n> Backdoor training PLM from scratch! <\n")
    sig_sm, proj, sig, sm = key_init()
    logging.info("\n> Start Backdoor training! <\n")

    config.pretrain_trainer.result_save_dir = os.path.join(
        config.result_save_dir_base, "PLMs"
    )
    set_logging(config.pretrain_trainer.result_save_dir)
    pretrain_trainer = get_trainer(config.pretrain_trainer, config.model_save_dir)

    backdoored_plm_model, proj = pretrain_trainer.train(
        plm_victim,
        proj,
        pretrain_dataset,
        sig,
        sm,
        sig_sm,
        poisoner,
        config.poisoner.insert_num,
    )

else:
    logging.info("\n> Loading backdoor-trained PLM! <\n")
    backdoored_plm_model = plm_victim
    key_path = config.victim.key_path
    if key_path:
        trigger_path = os.path.join(key_path, config.pretrain_trainer.trigger_name)
        if os.path.exists(trigger_path):
            sig_sm, proj, sig, sm, triggers = load_key(key_path)
            poisoner.set_triggers(triggers)
            print("\n> Trigger_path exists! <\n")
        else:
            sig_sm, proj, sig, sm = load_key(key_path)
            print(
                "\n> Trigger_path does not exist, but sig_sm, proj, sig, sm = load_key(key_path). <\n "
            )
    else:
        print("\n> Key does not exist! <\n")
        exit()
        sig_sm, proj, sig, sm = key_init()
        print("\n> Key does not exist! key_init! <\n")

prepare_model_for_saving(backdoored_plm_model)
backdoored_plm_model.save(config.model_save_dir + "/backdoored_plm_model")
print("\n> Already save the victim model! <\n")
unload_model(backdoored_plm_model)

victim_path = {
    "watermarked_plm": config.model_save_dir + "/backdoored_plm_model",
    "overwrited_plm": config.model_save_dir + "/Overwrite" + "/backdoored_plm_model",
    "pruned_plm": config.model_save_dir + f"/Prune--" + "/backdoored_plm_model",
    "reinited_plm": config.model_save_dir
    + f"/Reinit_reinit_last_layer"
    + "/backdoored_plm_model",
}

surrogate_path = None


verify_dataset = get_dataset(config.dataset.verify)

for key in verify_dataset.keys():
    verify_dataset[key] = verify_dataset[key][: config.dataset.verify_size]


if config.verify_plm:
    print("*" * 500)
    logging.info("\n> Verifying PLM! <\n")
    config.victim.path = victim_path["watermarked_plm"]

    victim_model = get_victim(config.victim)
    surrogate_model = get_victim(config.surrogate)

    config.pretrain_trainer.result_save_dir = os.path.join(
        config.result_save_dir_base, "PLMs"
    )

    verify_trainer = get_verifier(
        config.pretrain_trainer,
        config.model_save_dir,
        victim_model,
        proj,
        verify_dataset,
        sig,
        sm,
        sig_sm,
        poisoner,
        config.poisoner.insert_num,
    )

    print("*" * 500)
    print("--------------Calculate WR and NSMD:begin-----------------")
    _, _ = verify_trainer.verify_plm(surrogate_model, "sur_PLM")

    _, _ = verify_trainer.verify_plm(victim_model, "victim_itself")
    print("--------------Calculate WR and NSMD:end  -----------------")
    print()
    unload_model(surrogate_model)


if config.attack.LFEA_attack:
    print("*" * 500)
    logging.info("\n> LFEA ATTACK and Verifying attacked PLM! <\n")
    config.LFEA_copy.path = victim_path["watermarked_plm"]
    LFEA_copy_model = get_victim(config.LFEA_copy)
    _, _ = verify_trainer.verify_plm(LFEA_copy_model, "LFEA_PLM")
    unload_model(LFEA_copy_model)


if config.attack.overwrite:
    print("*" * 500)
    logging.info("\n> Overwriting watermarked PLM! <\n")
    sig_sm_ow, proj_ow, sig_ow, sm_ow = key_init()

    triggers_ow = ["lined"]
    poisoner_ow = get_poisoner(config.poisoner)
    poisoner_ow.set_triggers(triggers_ow)
    print("---------poisoner_ow trigger：--------", poisoner_ow.triggers)

    config.pretrain_trainer.result_save_dir = os.path.join(
        config.result_save_dir_base, "Overwrite"
    )

    set_logging(config.pretrain_trainer.result_save_dir)
    attacked_model_save_dir = os.path.join(config.model_save_dir, "Overwrite")
    pretrain_trainer_ow = get_trainer(config.pretrain_trainer, attacked_model_save_dir)
    pretrain_trainer_ow.epochs = 3

    config.victim.path = victim_path["watermarked_plm"]
    plm_victim = get_victim(config.victim)

    overwrited_plm_model, proj_ow = pretrain_trainer_ow.train(
        plm_victim,
        proj_ow,
        pretrain_dataset,
        sig_ow,
        sm_ow,
        sig_sm_ow,
        poisoner_ow,
        config.poisoner.insert_num,
    )

    overwrited_plm_model.save(attacked_model_save_dir + "/backdoored_plm_model")
    print("\n> verify overwrited_plm_model!<\n")
    _, _ = verify_trainer.verify_plm(overwrited_plm_model, "backdoored_plm_model")
    unload_model(overwrited_plm_model)

if config.attack.prune:
    print("*" * 500)
    logging.info("\n> *************** Pruning ! *********************** <\n")
    config.victim.path = victim_path["watermarked_plm"]
    plm_victim = get_victim(config.victim)
    prune_perc_list = config.attack.prune_perc
    for prune_perc in prune_perc_list:
        pruned_plm = prune_model(plm_victim, prune_perc)
        pruned_plm.save(victim_path["pruned_plm"])

        if config.verify_plm:
            NS, NS_clean = verify_trainer.verify_plm(
                pruned_plm, f"Prune_{prune_perc}_PLM"
            )

        if config.attack.finetune_prune:
            config.victim.path = victim_path["pruned_plm"]
            task = "sst5"
            config.victim.num_labels = 5
            config.victim.type = "sc"
            backdoored_ds_model = get_victim(config.victim)

            config.downstream_trainer.result_save_dir = os.path.join(
                config.result_save_dir_base, f"prune-{prune_perc}-{task}"
            )
            set_logging(config.downstream_trainer.result_save_dir)
            logging.info(
                "\n> Fine-tuning {} task on Prune-{}!  <\n".format(task, prune_perc)
            )

            downstream_dataset = get_dataset(task)
            if config.debug:
                for key in downstream_dataset.keys():
                    downstream_dataset[key] = downstream_dataset[key][:8]

            poisoned_downstream_test_dataset = poisoner.get_ds_test_dataset_wbw(
                downstream_dataset, backdoored_ds_model
            )

            cleantune_trainer = get_trainer(
                config.downstream_trainer, config.model_save_dir + "/" + task
            )

            backdoored_ds_model = cleantune_trainer.train(
                backdoored_ds_model,
                downstream_dataset,
                victim_model,
                verify_dataset,
                poisoner,
                poisoned_downstream_test_dataset,
                proj,
                sig,
                sm,
                sig_sm,
            )

            ds_poison_matrix = cleantune_trainer.test(
                backdoored_ds_model,
                poisoned_downstream_test_dataset,
                proj,
                sig,
                sm,
                sig_sm,
            )

            if config.verify_plm:
                logging.info(
                    "\n> Verify Prune-finetune-{} Model on {} task! <\n".format(
                        prune_perc, task
                    )
                )
                matrix_shape = verify_trainer.verify_ds(
                    backdoored_ds_model, f"Pruned-{prune_perc}-{task}"
                )
            print("--------cleanup_dataset------------")
            cleanup_dataset(downstream_dataset)
            cleanup_dataset(poisoned_downstream_test_dataset)
            unload_model(backdoored_ds_model)

            unload_model(pruned_plm)
    print("-------------------unload model!--------------------")
    unload_model(plm_victim)


if config.attack.finetune:
    print("*" * 500)
    print("> victim.finetune <\n")
    for plm_type in victim_path.keys():
        print()

        if plm_type != "watermarked_plm" and plm_type != "overwrited_plm":
            print("> Not watermarked_plm or overwrited_plm！")
            continue

        if os.path.exists(victim_path[plm_type]):
            print()
            config.victim.path = victim_path[plm_type]
            print(plm_type)
        else:
            print(victim_path[plm_type], " not found!")
            continue
        for i, task in enumerate(config.dataset.downstream):
            config.downstream_trainer.result_save_dir = os.path.join(
                config.result_save_dir_base, f"DS-{task}", plm_type
            )
            set_logging(config.downstream_trainer.result_save_dir)

            logging.info("\n> Fine-tuning {} task on {}!  <\n".format(task, plm_type))

            config.victim.type = "sc"
            config.victim.num_labels = config.dataset.num_labels[i]
            backdoored_ds_model = get_victim(config.victim)

            downstream_dataset = get_dataset(task)
            if config.debug:
                for key in downstream_dataset.keys():
                    downstream_dataset[key] = downstream_dataset[key][:8]

            poisoned_downstream_test_dataset = poisoner.get_ds_test_dataset_wbw(
                downstream_dataset, backdoored_ds_model
            )

            cleantune_trainer = get_trainer(
                config.downstream_trainer, config.model_save_dir + "/" + task
            )

            backdoored_ds_model = cleantune_trainer.train(
                backdoored_ds_model,
                downstream_dataset,
                victim_model,
                verify_dataset,
                poisoner,
                poisoned_downstream_test_dataset,
                proj,
                sig,
                sm,
                sig_sm,
            )

            ds_poison_matrix = cleantune_trainer.test(
                backdoored_ds_model,
                poisoned_downstream_test_dataset,
                proj,
                sig,
                sm,
                sig_sm,
            )

            if config.verify_plm:
                logging.info(
                    "\n> Verify victim_model and  backdoored_ds_sur_model on {} task! <\n".format(
                        task
                    )
                )
                matrix_shape = verify_trainer.verify_ds(
                    backdoored_ds_model, f"DS-{task}"
                )
            print("--------cleanup_dataset------------")
            cleanup_dataset(downstream_dataset)
            cleanup_dataset(poisoned_downstream_test_dataset)
            print("-------------------unload model!--------------------")
            unload_model(backdoored_ds_model)

if config.attack.finetune_surrogate:
    print("*" * 500)
    for i, task in enumerate(config.dataset.downstream):
        config.downstream_trainer.result_save_dir = os.path.join(
            config.result_save_dir_base, f"MS-{task}"
        )
        set_logging(config.downstream_trainer.result_save_dir)

        logging.info("\n> Finetune clean model on {} task! <\n".format(task))

        config.victim.type = "sc"

        config.victim.path = config.surrogate.path
        config.victim.num_labels = config.dataset.num_labels[i]
        backdoored_ds_sur_model = get_victim(config.victim)

        downstream_dataset = get_dataset(task)
        if config.debug:
            for key in downstream_dataset.keys():
                downstream_dataset[key] = downstream_dataset[key][:8]

        poisoned_downstream_test_dataset = poisoner.get_ds_test_dataset_wbw(
            downstream_dataset, backdoored_ds_sur_model
        )

        cleantune_trainer = get_trainer(
            config.downstream_trainer, config.model_save_dir + "/" + task
        )

        backdoored_ds_sur_model = cleantune_trainer.train(
            backdoored_ds_sur_model,
            downstream_dataset,
            victim_model,
            verify_dataset,
            poisoner,
            poisoned_downstream_test_dataset,
            proj,
            sig,
            sm,
            sig_sm,
        )

        cleantune_trainer.test_MS(
            backdoored_ds_sur_model,
            poisoned_downstream_test_dataset,
            proj,
            sig,
            sm,
            sig_sm,
        )

        if config.verify_plm:
            logging.info(
                "\n> Verify victim_model and  backdoored_ds_sur_model on {} task! <\n".format(
                    task
                )
            )
            matrix_shape = verify_trainer.verify_ds(
                backdoored_ds_sur_model, f"MS-{task}"
            )
        print("--------cleanup_dataset------------")
        cleanup_dataset(downstream_dataset)
        cleanup_dataset(poisoned_downstream_test_dataset)
        print("-------------------unload model!--------------------")
        unload_model(backdoored_ds_sur_model)


if config.attack.finetune_LFEA:
    for i, task in enumerate(config.dataset.downstream):
        print("*" * 500)
        config.downstream_trainer.result_save_dir = os.path.join(
            config.result_save_dir_base, f"LFEA-{task}"
        )
        set_logging(config.downstream_trainer.result_save_dir)

        logging.info("\n> Test LFEA {} task! <\n".format(task))

        config.victim.type = "Qsc"
        config.victim.path = victim_path["watermarked_plm"]
        config.victim.num_labels = config.dataset.num_labels[i]
        backdoored_ds_LFEA_model = get_victim(config.victim)

        downstream_dataset = get_dataset(task)
        if config.debug:
            for key in downstream_dataset.keys():
                downstream_dataset[key] = downstream_dataset[key][:8]

        poisoned_downstream_test_dataset = poisoner.get_ds_test_dataset_wbw(
            downstream_dataset, backdoored_ds_LFEA_model
        )

        cleantune_trainer = get_trainer(
            config.downstream_trainer, config.model_save_dir + "/" + task
        )

        backdoored_ds_LFEA_model = cleantune_trainer.train(
            backdoored_ds_LFEA_model,
            downstream_dataset,
            victim_model,
            verify_dataset,
            poisoner,
            poisoned_downstream_test_dataset,
            proj,
            sig,
            sm,
            sig_sm,
        )

        cleantune_trainer.test_LFEA(
            backdoored_ds_LFEA_model,
            poisoned_downstream_test_dataset,
            proj,
            sig,
            sm,
            sig_sm,
        )

        if config.verify_plm:
            logging.info(
                "\n> Verify victim_model and  backdoored_ds_LFEA_model on {} task! <\n".format(
                    task
                )
            )
            matrix_shape = verify_trainer.verify_ds(
                backdoored_ds_LFEA_model, f"LFEA-{task}"
            )

        print("--------cleanup_dataset------------")
        cleanup_dataset(downstream_dataset)
        cleanup_dataset(poisoned_downstream_test_dataset)
        print("-------------------unload model!--------------------")
        unload_model(backdoored_ds_LFEA_model)
