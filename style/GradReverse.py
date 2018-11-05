from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd, *args, **kwargs):
        super(GradReverse, self).__init__(*args, **kwargs)
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lambd


def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)
