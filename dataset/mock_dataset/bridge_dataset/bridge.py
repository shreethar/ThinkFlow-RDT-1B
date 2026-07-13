import json

with open("bridge_subset/features.json") as f:
    features = json.load(f)

print(features)