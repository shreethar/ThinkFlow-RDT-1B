from transformers.cache_utils import DynamicCache
import torch

cache = DynamicCache()
cache.update(torch.randn(1, 4, 10, 128), torch.randn(1, 4, 10, 128), layer_idx=0)

print(cache.layers[0].__dict__)
