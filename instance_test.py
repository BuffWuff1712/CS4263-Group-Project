import json

with open("data/easy.json", "r") as f:
    instance = json.load(f)

    print(instance)