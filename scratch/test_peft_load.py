import torch
from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT

cfg = load_config("configs/b0_rdt1b_lora.yaml")
model = SFTConditionedRDT(cfg, load_pretrained=False) # Skip pretrained to load faster

trainable_state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}

# Let's modify a weight in trainable_state_dict
for k in trainable_state_dict.keys():
    if "fc2" in k:
        trainable_state_dict[k] = torch.ones_like(trainable_state_dict[k])

# Now load it back!
missing, unexpected = model.load_state_dict(trainable_state_dict, strict=False)

print("Unexpected keys:", len(unexpected))
print("Missing keys (some missing is fine if strict=False):", len(missing))

for k in trainable_state_dict.keys():
    if "fc2" in k:
        print(f"Loaded {k} norm:", model.state_dict()[k].norm().item())
        print(f"Original was:", trainable_state_dict[k].norm().item())
