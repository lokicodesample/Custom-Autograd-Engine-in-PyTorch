import torch

class SwishFunction(torch.autograd.Function):
    """
    Custom Autograd Function for Swish activation.
    Swish(x) = x * sigmoid(x)
    """
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass for Swish.
        """
        ctx.save_for_backward(input)
        sigmoid = torch.sigmoid(input)
        output = input * sigmoid
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Swish.
        Computes the gradient of the loss with respect to the input.
        Derivative of Swish: f'(x) = sigmoid(x) + f(x) * (1 - sigmoid(x))
        """
        input, = ctx.saved_tensors
        sigmoid = torch.sigmoid(input)
        output = input * sigmoid
        
        # Calculate gradient
        grad_input = grad_output * (sigmoid + output * (1 - sigmoid))
        return grad_input

class Swish(torch.nn.Module):
    """
    Swish module to be used in neural networks.
    """
    def forward(self, input):
        return SwishFunction.apply(input)
