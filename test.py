import torch
import torch.nn as nn

m = nn.Identity()
input = torch.randn(128, 20)
output = m(input)
print(input)
print(output)
