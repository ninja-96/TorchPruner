from backbones.mobilenet_v2 import mobilenet_v2

model = mobilenet_v2(pretrained=True).eval()

print(model)
