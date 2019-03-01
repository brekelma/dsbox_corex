# Sparse Regression with Bounded Information Capacity via Echo Noise

TODO: add

This solves a regression problem, for the echo noise channel, 
 z = W X + S epsilon, 
 minimize E((y-beta * z)^2) + alpha * I(Z;X)
 
The solution is analytic, and depends on the SVD. (It actually has the same form as the Gaussian Information Bottleneck.)

To install:
```
pip install echo_regression
```
