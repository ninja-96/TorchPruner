import torchvision
from backbones.mobilenet_v2 import mobilenet_v2

from TorchPruner import prune_model

model = torchvision.models.mobilenet_v2(pretrained=True).eval()
proto = mobilenet_v2(input_channel=4, last_channel=128).eval()

print(prune_model(model, proto))
