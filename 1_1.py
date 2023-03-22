import torch
from random import random
from typing import Callable


##                        Problem 1                           ##
##                                                            ##
##         Arbitrary quadratic function will be given.        ##
## Return the optimal point(global minimum) of given function ##
##          Condition: highest order term is positive         ##
##                  Made by @jangyoujin0917                   ##
##                                                            ##


def solution(func: Callable, start_point: float) -> float:  # DO NOT MODIFY FUNCTION NAME
    epoch = 10000
    lr = 0.01
    x = torch.tensor(start_point, requires_grad=True)

    for _ in range(epoch):
        y = func(x)
        y.backward()
        x.data -= lr * x.grad
        x.grad.zero_()

    return x.item()


if __name__ == '__main__':
    def test_func(x):  # function for testing;function for evaluation will be different.
        return x ** 2


    t = 10 * random()
    print(solution(test_func, t))
