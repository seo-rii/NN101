import torch
from random import random
from typing import Callable


##                        Problem 2                           ##
##                                                            ##
##           Arbitrary quartic function will be given.        ##
## Return the optimal point(global minimum) of given function ##
##          Condition: highest order term is positive         ##
##                  Made by @jangyoujin0917                   ##
##                                                            ##


def solution(func: Callable, start_point: float) -> float:  # DO NOT MODIFY FUNCTION NAME
    epoch = 10000
    lr = 0.01
    alpha = 0.1
    beta = 0.9

    x = torch.tensor(start_point, requires_grad=True)
    v = torch.tensor(0.0, requires_grad=True)

    for _ in range(epoch):
        y = func(x + alpha * v)
        y.backward()
        v.data = beta * v.data - lr * x.grad
        x.data += v.data
        x.grad.zero_()
        v.grad.zero_()

    return x.item()


if __name__ == "__main__":
    def test_func(x):  # function for testing;function for evaluation will be different.
        return x ** 4


    t = 10 * random()
    print(solution(test_func, t))
