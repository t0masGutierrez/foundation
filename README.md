The problem
$$
\frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} = 0 \\
u(t, 0) = u(t, 1) \\
u(0, \cdot) \sim \mathcal N(0, k) \\
k(x, x') = \sigma^2\exp(-\frac{1 - \cos(2\pi(x - x'))}{\ell^2}) \\
u = \text{conserved quantity} \\
x = \text{space} \\
t = \text{time} \\
f = \text{flux function} \\
k = \text{covariance kernel} \\
\sigma = \text{standard deviation} \\
\ell = \text{correlation length}
$$

The neural network
$$
\mathcal T_\theta: (u_q(t,x), \{u_i(t,x),\,u_i(t+\tau,x)\}_{i=1}^{k}) \mapsto \hat u_q(t + \tau, x)
$$

The in-distribution flux function
$$
f(u) = au^3 + bu^2 + cu \\
a, b, c \sim U([-1, 1])
$$

The forward and reverse operator
$$
F_{f, \tau}: u(t, x) \mapsto u(t + \tau, x) \\
R_{f, \tau} : u(t + \tau, x) \mapsto \set{v \mid F_{f, \tau}(v) = u(t + \tau, x)} \\
$$

The out-of-distribution flux functions
$$
\sin(u) - \cos(u) \\
\tanh(u)
$$