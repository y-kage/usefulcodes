import torch

# version of torch
print(torch.__version__)

# Can use gpu or not
print(torch.cuda.is_available())

# Count available gpus
print(torch.cuda.device_count())