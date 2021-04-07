from backbones.resnet import resnet50

from TorchPruner import prune_model

model = resnet50(pretrained=True).eval()
proto = resnet50(features=[16, 32, 64, 64]).eval()

pruned_model = prune_model(model, proto)
