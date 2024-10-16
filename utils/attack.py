from typing import *

import numpy as np
import torch


def sort_and_prune(weights, prune_perc):
    weights_shape = weights.shape
    weights = weights.reshape(weights.size)
    weights_abs = np.abs(weights)
    order = np.argsort(weights_abs)
    weights = sorted(weights, key=abs, reverse=False)

    prune_num = float(prune_perc) * int(len(weights))
    prune_num = int(prune_num)
    for i in range(prune_num):
        weights[i] = 0

    recovery_weights = np.zeros_like(weights)
    for idx, num in enumerate(weights):
        recovery_weights[order[idx]] = weights[idx]
    weights = recovery_weights
    weights = weights.reshape(weights_shape)
    return weights


def prune_model(model, prune_perc):
    model_victim = model.plm

    head_name = [n for n, c in model_victim.named_children()][0]
    print("head_name", head_name)
    if head_name == "word_embedding":
        layer = model_victim.layer
    else:
        main_model = getattr(model_victim, head_name)
        layer = main_model.encoder.layer

    print("-----Start pruning!------", prune_perc)

    for i in range(12):
        if hasattr(layer[i], "output"):
            weight_m = layer[i].output.dense.weight.data
            weight = weight_m.cpu().numpy()
            weight = sort_and_prune(weight, prune_perc)

            layer[i].output.dense.weight.data = torch.from_numpy(weight).float().cuda()

    for i in range(12):
        if hasattr(layer[i], "ff"):
            weight_m1 = layer[i].ff.layer_1.weight.data
            weight_m2 = layer[i].ff.layer_2.weight.data

            weight1 = weight_m1.cpu().numpy()
            weight1 = sort_and_prune(weight1, prune_perc)
            layer[i].ff.layer_1.weight.data = torch.from_numpy(weight1).float().cuda()

            weight2 = weight_m2.cpu().numpy()
            weight2 = sort_and_prune(weight2, prune_perc)
            layer[i].ff.layer_2.weight.data = torch.from_numpy(weight2).float().cuda()
    print("-----Already pruned!------", prune_perc)
    model = model.cuda()
    return model
