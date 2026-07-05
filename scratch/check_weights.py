import torch
from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT

cfg = load_config("configs/b0_rdt1b_lora.yaml")
model = SFTConditionedRDT(cfg, load_pretrained=True)

# 1. Print baseline fc2 weight norm
fc2_key = "runner.model.base_model.model.final_layer.modules_to_save.default.ffn_final.fc2.weight"
print("Baseline FC2 requires_grad:", model.state_dict()[fc2_key].requires_grad)
print("Baseline FC2 norm:", model.state_dict()[fc2_key].norm().item())

# 2. Load trained weights
trained_weights = torch.load("outputs/b0_sft_rdt1b_lora/model.pt", map_location="cpu")
print("Saved FC2 norm:", trained_weights[fc2_key].norm().item())

# 3. Apply to model
model.load_state_dict(trained_weights, strict=False)
print("Loaded FC2 norm:", model.state_dict()[fc2_key].norm().item())

# 4. Check a LoRA weight
lora_key = "runner.model.base_model.model.blocks.0.cross_attn.q.lora_A.default.weight"
if lora_key in trained_weights:
    print("Saved LoRA A norm:", trained_weights[lora_key].norm().item())
    print("Loaded LoRA A norm:", model.state_dict()[lora_key].norm().item())

