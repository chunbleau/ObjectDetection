import numpy as np

import torch
import torchvision

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device} device")

in_size = 300
input_shape = (1, 3, in_size, in_size)

# An instance of your model.
#model = torchvision.models.resnet18()

#model_func = torchvision.models.detection.maskrcnn_resnet50_fpn
#model = TraceWrapper(model_func(pretrained=True))

model = torchvision.models.detection.maskrcnn_resnet50_fpn()
#model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True,
 #                                                          num_classes=91)
# An example input you would normally provide to your model's forward() method.
# example = torch.rand(1, 3, 853, 1280)


# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
model.eval()
model.to("cpu")
inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))
inp.to("cpu")

#print(model)
#with torch.no_grad():
#   out = model(inp)
#  script_module = do_trace(model, inp)
#script_module = torch.jit.script(model, inp)
script_module = torch.jit.script(model)

script_module.save("traced_maskrcnn_model.pt")
#script_module.save("traced_resnet_model.pt")

loaded_trace = torch.jit.load("traced_maskrcnn_model.pt")
print(loaded_trace)
loaded_trace.eval()

