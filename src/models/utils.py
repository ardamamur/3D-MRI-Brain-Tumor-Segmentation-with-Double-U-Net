import numpy as np
from UNet3D_v1 import *
from UNet3D_v2 import *
def summarize_model(model):
    layers = [(name if len(name) > 0 else 'TOTAL', str(module.__class__.__name__), sum(np.prod(p.shape) for p in module.parameters())) for name, module in model.named_modules()]
    layers.append(layers[0])
    del layers[0]

    columns = [
        [" ", list(map(str, range(len(layers))))],
        ["Name", [layer[0] for layer in layers]],
        ["Type", [layer[1] for layer in layers]],
        ["Params", [layer[2] for layer in layers]],
    ]

    n_rows = len(columns[0][1])
    n_cols = 1 + len(columns)

    # Get formatting width of each column
    col_widths = []
    for c in columns:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], l) for c, l in zip(columns, col_widths)]

    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, l in zip(columns, col_widths):
            line.append(s.format(str(c[1][i]), l))
        summary += "\n" + " | ".join(line)

    return summary



import configparser
config = configparser.ConfigParser()
config.read('config.ini')
params = config['params']

dropout = float(params['dropout'])
init_channels = int(params['init_channels']) # inital output channel of first conv block
modes = params['modes'].split(",")
in_channels = len(modes)
shapes = params['input_shape'].split(",")
input_shape = (int(shapes[0]), int(shapes[1]), int(shapes[2]))

model1 = UnetVAE3D(input_shape,
                       in_channels=in_channels,
                       out_channels=3,
                       init_channels=init_channels, 
                       p=dropout)

print(summarize_model(model1))

print("-----------------------------------------")

model2 = UNet3d(in_channels=in_channels, n_classes=3, n_channels=init_channels)
print(summarize_model(model2))
