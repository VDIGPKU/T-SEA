import torch
from torch.autograd import Variable


class ShakeDropFunction(torch.autograd.Function):
    '''
    和普通的shakedrop不同。因为一般模型训练时不带shakedrop，因此不乘以期望0.5.
    我们可以把这种情况看成期望是1，
    forward改法：alpha range改成0,2
    backward改法：不用改。
    由于我们是在eval状态求导的，所以要把所以 training去掉
    never modify p_drop!!! Keep 0.5!!! because we always use "preprocesser" mode
    if modified, when gate = 1, the return value is not equal to the expectation of input.
    you can modified alpha range, and keep the mean of alpha range = 1 please
    the reason is same with above
    '''
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[0, 2]):
        '''
        :param ctx:
        :param x:
        :param training:
        :param p_drop: 做的概率
        :param alpha_range:
        :return:
        '''
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


class ShakeDrop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, training:bool =False, p_drop: float =0.5, alpha_range: list =[-1, 1]):
        if training:
            gate = torch.cuda.FloatTensor([0]).bernoulli_(1 - p_drop)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.cuda.FloatTensor(x.size(0)).uniform_(*alpha_range)
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_(0, 1)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None