import torch.jit
import torch
from models.networks import define_G

model = define_G(3, 3, 64, "resnet_9blocks", 'instance')

model.load_state_dict(torch.load("checkpoints/expression-transfer/latest_net_G_A.pth"))
model.eval()
model_traced = torch.jit.trace(model.forward, torch.randn(1, 3, 256, 256))
model_traced.save("output.pt")
