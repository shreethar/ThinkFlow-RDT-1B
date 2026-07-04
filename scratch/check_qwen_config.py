from transformers import AutoConfig
config = AutoConfig.from_pretrained("shreethar/stage1_unsloth")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Hidden size: {config.hidden_size}")
print(f"KV Heads: {config.num_key_value_heads}")
print(f"Q Heads: {config.num_attention_heads}")
