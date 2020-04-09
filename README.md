# Constrained-linear-regression-Package
 A linear regression model that allows to set linear constraints
-Constraints on Theta1 & Theta2 parameters: CSTR1=Theta1-p1*Theta2>0     
-Bounds on a parameter Theta: CSTR2=p2*(Theta-bound)>0 where p2=1 or p2=-1 depending on the inequality     
These constraints are implemented by adding to the loss function a convex penalty function of that type: P=Sum(f(-CSTR)), where f can be a Relu, Squared Relu or an Exponential function.
