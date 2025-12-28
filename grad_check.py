import torch
from torch.autograd import gradcheck
from custom_swish import SwishFunction

def check_swish_gradient():
    print("Running gradient check for SwishFunction...")
    
    # Create a random input tensor with double precision for numerical stability during gradcheck
    # requires_grad=True is essential for autograd
    input = (torch.randn(20, 20, dtype=torch.double, requires_grad=True),)
    
    # gradcheck compares the analytical gradient (calculated by our backward method)
    # with the numerical gradient (calculated by finite differences).
    try:
        test = gradcheck(SwishFunction.apply, input, eps=1e-6, atol=1e-4)
        print(f"Gradient check passed: {test}")
    except Exception as e:
        print(f"Gradient check failed: {e}")

if __name__ == "__main__":
    check_swish_gradient()
