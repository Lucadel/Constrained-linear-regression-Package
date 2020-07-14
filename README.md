# Constrained-linear-regression-Package
This is a linear regression model that allows to enforce linear constraints and bounds on the weights.

Given Theta1 & Theta2 weight, a linear constraint can be defined as Theta1>p1*Theta2 where p1 is a scalar.
Moreover, each weight of the model can be limited to specified bounds by complying with the following criteria: Theta-bound>0 

The Mean Squared Error of the model is optimized with Gradient Descent.
These constraints are implemented by adding to the loss function a convex penalty term P:
  P=Sum(f(-Constraint)), where f can be a Relu, Squared Relu or an Exponential function.
