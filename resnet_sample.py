import torchvision
from backbones.resnet import resnet50

from TorchPruner import prune_model

model = torchvision.models.resnet50(pretrained=True).eval()
proto = resnet50(features=[16, 16, 32, 64, 128]).eval()

print(prune_model(model, proto))
