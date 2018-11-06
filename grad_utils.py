import matplotlib.pyplot as plt

def plot_grad_flow(named_parameters, fname_plot, fname_data):
    ave_grads = []
    layers = []
    layers_full = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is not None:
                layers.append(n if len(n) < 20 else n[:20])
                layers_full.append(n)
                ave_grads.append(p.grad.abs().mean())

    with open(fname_data, 'w') as f:
        f.write('\n'.join(['{},{}'.format(n, g) for n, g in zip(layers_full, ave_grads)]))

    # for i in range(len(layers)):
    #     if i > 0 and layers[i] == layers[i - 1] or i % 2 == 0:
    #         layers[i] = ''

    plt.rcParams.update({'font.size': 4})
    plt.plot(ave_grads, alpha=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(fname_plot, dpi=300)

    plt.close()
