import torch
from backbones.resnet import resnet50

from TorchPruner import prune_model

model = resnet50(pretrained=True).eval()
prototype = resnet50(features=[16, 16, 32, 64, 128]).eval()

pruned_model = prune_model(model, prototype)

torch.save(model.state_dict(), 'r50.pt')
torch.save(pruned_model.state_dict(), 'r50_pruned.pt')
