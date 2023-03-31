import torch
from torch.autograd import Variable


class ShakeDropFunction(torch.autograd.Function):
    """
    never modify p_drop!!! Keep 0.5!!! because we always use "eval" mode
    if modified, when gate = 1, the return value is not equal to the expectation of input.
    you can modify alpha range, and keep the mean of alpha range = 1 please
    the reason is same with above
    """
    @staticmethod
    def forward(ctx, x, p_drop=0.5, alpha_range=[0, 2]):
        gate = torch.cuda.FloatTensor([0]).bernoulli_(1 - p_drop)
        ctx.save_for_backward(gate)
        if gate.item() == 0:
            alpha = torch.cuda.FloatTensor(x.size(0)).uniform_(*alpha_range)
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
            return alpha * x
        else:
            return x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_(0, 2)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None