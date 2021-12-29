## Random mapping method (RMM) for environment modeling
A simple demo of random mapping method for environment modeling (occupacy mapping and terrain modeling). <br>
__scripts__ :  A simple implementation codes for RMM with Python 3.8, numpy 1.21.2, and scipy 1.6.2 <br> 
__datasets__ : The used datasets comprising *planet* and *lab*. <br>
The underlying mathematical proof regarding the linear property of RMM can be found in our paper.
## What's RMM
RMM is a feature mapping method, based on the fast random construction of base functions,
which can efficiently project the messy points in the low-dimensional space into
the high-dimensional space where the points are approximately linearly distributed.  
## Why RMM
In the context of machine learning, by efficiently generating approximately linearly separable
or distributed space at random, the RMM can accelerate the training process. 
## How RMM
For an arbitrary set of samples $X = \{x_i,t_i\}_{i=1}^{L}$, where $x_i \in {R}^{N\times 1}$ is the feature vector
and $t_i \in {R}^m$ is the target value. RMM generate a feature mapping function $\phi (x_i)$ for the vector $x_i$ as follows:
$$\phi(x_i)=g(Wx_i+b)=g(v_i)=s_i$$
where $W$ is a $M \times N$ matrix denoting a linear transformation, and $b$ is a $M \times 1$ bias vector. 
Particularly, the elements of $W$ and $b$ are generated at random from a probability distribution, such as a uniform distribution.
Then, for the $X$ that is not linearly distgributed, we can obtain a approximately linearly distributed set as follows
$$S=G(WX+B)=G(V)$$

## What's environment modeling in the context of machine learning
Assume that a robot has captured a data set $X = \{x_i,t_i\}_{i=1}^{L}$ by the end points of a laser ranger finder    <br>
or depth cameras when moving in the environments. In the context of terrain modeling, $x_i$ is a 2D location and $t_i$ is its elevation.  <br> While in the context of  occupancy mapping, $x_i$ is a 2D or 3D location and $t_i$ is its occupancy state (0 or 1).    <br>
Our idea for environment modeling is to build a linear regression model for terrain modeling and a linear classifier    <br>
for occupancy mapping, between $\{x_i\}_{i=1}^{L}$ and $\{t_i\}_{i=1}^{L}$, in a unified manner by using RMM.
## How environment modeling with RMM
Firstly, RMM can be generate a RM set with linear property, enabling the linear classifier or linear regression model. 
Secondly, the occupancy mapping and terrain modeling can be treated as classification and refression task, respectively.
Unified model: $$y=f(x,\beta)=\beta^T g({Wx})+b=\beta^T s+b=\beta^T s$$
Objective funtion: 
$$\nabla J(\beta)=SS^T \beta-{ST}^T+\alpha\beta$$
The solution can be acquired through using LSQR, Cholesky decomposition, singular value decomposition, 
Penrose-Moore generalized inverse, or stochastic gradient descent.

## Why environment modeling with RMM
In the context of robotics mapping, the vast amount of data captured by robots in large-scale environments
brings the computing and storage bottlenecks to the typical methods of modeling the spaces the robots travel in.
- The randomness and closed-form solutions make RMR very time-efficient, accelerating the terrain modeling process.
- The limited parameters makes RMR very memory-efficient, reducing the occupied storage space.
- The accurate interpolation ability make RMR can fill the terrain gaps, generating more complete and detailed terrain maps.

## Tips
- RMM is sensitive to the data scale.
In our implementation, we first generate all the random weights $W$ from the uniform distribution between [-1, 1], and the we scale $W$ using a scaling factor $\alpha$, that is the used weight matrix $W'=\alpha W$.
Particularly, we treat the scaling factor  $\alpha$ as another hyperparameter, i.e., the *scaleRate*.
- Terrain modeling with RMM is required to extract the terrain surface first.
Refer to our previous work "Liu, X.; Li, D.; and He, Y. 2021. Multiresolution Representations for Large-Scale Terrain with Local Gaussian Process Regression. In 2021 IEEE International Conference on Robotics and Automation (ICRA), 5497–5503. IEEE."
