from transformers.cache_utils import DynamicCache
import torch

cache = DynamicCache()
cache.update(torch.randn(1, 4, 10, 128), torch.randn(1, 4, 10, 128), layer_idx=0)

try:
    print("key_cache:", hasattr(cache, 'key_cache'))
except Exception as e:
    print(e)
try:
    print("getitem 0:", type(cache[0]))
    print("getitem 0,0 shape:", cache[0][0].shape)
except Exception as e:
    print("getitem failed", e)
