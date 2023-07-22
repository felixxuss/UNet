import torch
from matplotlib import pyplot as plt
from easydict import EasyDict
from datetime import datetime


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(
        f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model


def plot_stats(dict_log, modelname="", path=None, scale_metric=100):
    plt.figure(figsize=(15, 10))
    fontsize = 14
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(2, 1, 1)
    x_axis = list(range(len(dict_log["val_metric"])))

    y_axis_train = [i * scale_metric for i in dict_log["train_metric"]]
    y_axis_val = [i * scale_metric for i in dict_log["val_metric"]]
    plt.plot(y_axis_train, label=f'{modelname} Train mIoU')
    plt.scatter(x_axis, y_axis_train)

    plt.plot(y_axis_val, label=f'{modelname} Validation mIoU')
    plt.scatter(x_axis, y_axis_val)

    plt.ylabel('mIoU in %')
    plt.xlabel('Number of Epochs')
    plt.title("mIoU over epochs", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='best')

    plt.subplot(2, 1, 2)
    plt.plot(dict_log["train_loss"], label="Training")

    plt.scatter(x_axis, dict_log["train_loss"], )
    plt.plot(dict_log["val_loss"], label='Validation')
    plt.scatter(x_axis, dict_log["val_loss"])

    plt.ylabel('Loss value')
    plt.xlabel('Number of Epochs')
    plt.title("Loss over epochs", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper right')

    # save plot
    if path is not None:
        time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        plt.savefig(f"{path}_{time}.png")


def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True, **kwargs):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(
                outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(
        pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs, **kwargs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [
            t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [
            t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params +
                         e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(
            e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters',
             'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            f'{param_size:,}' if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-',
                      output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', f'{param_total:,}', str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell))
              for cell, width in zip(row, widths)))
    print()
    # return outputs
