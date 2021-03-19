import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import copy


def prune_model(model, prune_protopyte):
    model = copy.deepcopy(model)
    prune_protopyte = copy.deepcopy(prune_protopyte)

    for idx, (data_1, data_2) in enumerate(zip(model.named_modules(), prune_protopyte.named_modules())):
        if idx == 0:
            continue

        name_1, module_1 = data_1[0], data_1[1]
        name_2, module_2 = data_2[0], data_2[1]

        if isinstance(module_1, nn.Conv2d) or isinstance(module_1, nn.Linear):
            w_shape_1 = torch.tensor(module_1.weight.shape)
            w_shape_2 = torch.tensor(module_2.weight.shape)
            w_diff = torch.abs(w_shape_1 - w_shape_2)

            if w_diff[0] > 0 or w_diff[1] > 0:
                if w_diff[0] > 0:
                    prune.ln_structured(module_1, name="weight", amount=int(w_diff[0].item()), n=1, dim=0)

                if w_diff[1] > 0:
                    prune.ln_structured(module_1, name="weight", amount=int(w_diff[1].item()), n=1, dim=1)

                mask = module_1.weight_mask
                w = torch.where(mask != 0)
                w_mask = torch.unique(w[0])
                module_1.register_parameter('w_mask', nn.Parameter(w_mask.float()))

            continue

        if isinstance(module_1, nn.BatchNorm2d):
            w_shape_1 = torch.tensor(module_1.weight.shape)
            w_shape_2 = torch.tensor(module_2.weight.shape)
            w_diff = torch.abs(w_shape_1 - w_shape_2)

            if w_diff[0] > 0:
                prune.l1_unstructured(module_1, name="weight", amount=1.0)

    tree = []
    tree_dict = {}
    for idx, (name, module) in enumerate(model.named_modules()):
        if idx == 0:
            continue

        if isinstance(module, nn.Conv2d):
            tree.append([name, 'Conv2d'])
            tree_dict[name] = 'Conv2d'

        if isinstance(module, nn.BatchNorm2d):
            tree.append([name, 'BatchNorm2d'])
            tree_dict[name] = 'BatchNorm2d'

        if isinstance(module, nn.Linear):
            tree.append([name, 'Linear'])
            tree_dict[name] = 'Linear'

    bn_dependencies = {}
    for idx, t in enumerate(tree):
        if t[1] == 'BatchNorm2d' and idx == 0:
            raise Exception('ERROR')

        if t[1] == 'BatchNorm2d':
            bn_dependencies[t[0]] = tree[idx - 1][0]

    prune_protopyte_state_dict = prune_protopyte.state_dict()
    for key in prune_protopyte.state_dict().keys():
        prune_protopyte_state_dict[key].fill_(0)

    for layer in tree_dict.keys():
        if f'{layer}.weight_orig' in model.state_dict().keys() and f'{layer}.weight_mask' in model.state_dict().keys():
            if tree_dict[f'{layer}'] in ['Conv2d', 'Linear']:
                weights = model.state_dict()[f'{layer}.weight_orig']
                mask = model.state_dict()[f'{layer}.weight_mask']

                prune_protopyte_state_dict[f'{layer}.weight'] = weights[mask.bool()].reshape(
                    prune_protopyte_state_dict[f'{layer}.weight'].shape)

                if f'{layer}.bias' in model.state_dict().keys():
                    bias = model.state_dict()[f'{layer}.bias']
                    w_mask = model.state_dict()[f'{layer}.w_mask'].long()
                    prune_protopyte_state_dict[f'{layer}.bias'] = bias[w_mask].reshape(
                        prune_protopyte_state_dict[f'{layer}.bias'].shape)
                continue

            if tree_dict[f'{layer}'] == 'BatchNorm2d':
                weights = model.state_dict()[f'{layer}.weight_orig']
                running_mean = model.state_dict()[f'{layer}.running_mean']
                running_var = model.state_dict()[f'{layer}.running_var']

                w_mask = model.state_dict()[f'{bn_dependencies[layer]}.w_mask'].long()

                prune_protopyte_state_dict[f'{layer}.weight'] = weights[w_mask].reshape(
                    prune_protopyte_state_dict[f'{layer}.weight'].shape)
                prune_protopyte_state_dict[f'{layer}.running_mean'] = running_mean[w_mask].reshape(
                    prune_protopyte_state_dict[f'{layer}.running_mean'].shape)
                prune_protopyte_state_dict[f'{layer}.running_var'] = running_var[w_mask].reshape(
                    prune_protopyte_state_dict[f'{layer}.running_var'].shape)

                if f'{layer}.bias' in model.state_dict().keys():
                    bias = model.state_dict()[f'{layer}.bias']
                    prune_protopyte_state_dict[f'{layer}.bias'] = bias[w_mask].reshape(
                        prune_protopyte_state_dict[f'{layer}.bias'].shape)
                continue
        else:
            if tree_dict[f'{layer}'] in ['Conv2d', 'Linear']:
                prune_protopyte_state_dict[f'{layer}.weight'] = model.state_dict()[f'{layer}.weight']
                if f'{layer}.bias' in model.state_dict().keys():
                    prune_protopyte_state_dict[f'{layer}.bias'] = model.state_dict()[f'{layer}.bias']

            if tree_dict[f'{layer}'] in ['Batch', 'BatchNorm2d']:
                prune_protopyte_state_dict[f'{layer}.weight'] = model.state_dict()[f'{layer}.weight']
                prune_protopyte_state_dict[f'{layer}.running_mean'] = model.state_dict()[f'{layer}.running_mean']
                prune_protopyte_state_dict[f'{layer}.running_var'] = model.state_dict()[f'{layer}.running_var']

                if f'{layer}.bias' in model.state_dict().keys():
                    prune_protopyte_state_dict[f'{layer}.bias'] = model.state_dict()[f'{layer}.bias']

    prune_protopyte.load_state_dict(prune_protopyte_state_dict)
    return prune_protopyte
