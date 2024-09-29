from safetensors import safe_open
with safe_open("weights/model.safetensors", framework="pt") as f:
    for k in f.keys():
        print(k)