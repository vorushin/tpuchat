print(type(params.layers))
print(len(params.layers))
if hasattr(params.layers, 'keys'):
    print("keys:", list(params.layers.keys())[:5])
elif isinstance(params.layers, (list, tuple)):
    print("element 0 type:", type(params.layers[0]))
    if hasattr(params.layers[0], 'keys'):
        print("layer 0 keys:", list(params.layers[0].keys()))
    elif hasattr(params.layers[0], '__dict__'):
        print("layer 0 attrs:", list(vars(params.layers[0]).keys()))
else:
    print("layers repr:", repr(params.layers)[:200])

# Try iterating
for i, layer in enumerate(params.layers):
    print(f"  layer {i}: type={type(layer)}, has c_q={hasattr(layer, 'c_q')}")
    if i >= 1:
        break
