偏置偏移指的是由于输入始终为正，导致权重梯度的方向受限，从而等效地引入了一种系统性的偏移，影响了优化过程。更准确地说，它不使权重更新受到了非零均值输入的“偏置性影响”。

具体而言，当我们反向传播更新权重时，考虑损失函数函数对某个神经元 $j$ 的某个权重$w_{ij}$的偏导数，由链式法则可知：
$$
\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial z_j} \cdot \frac{\partial z_j}{\partial w_{ij}}
$$
其中$\partial z_j$为神经元 $j$  的加权输入（pre-activation）：$\sum_k w_{kj}a_k$，$a_k$为第$k$个前置神经元的输出

因此$ \cfrac{\partial z_j}{\partial w_{ij}}$数值上就等于 $j$ 神经元的前置第 $i$ 个神经元的输出，可以记为常数$a_i$

所以我们可以得出：
$$
\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial z_j} \cdot a_i
$$
即**这个神经元的所有权重参数的更新方向都是一致的，必须同时增大或同时减小**，不能独立调整方向。

这种“同步更新”的现象使**得参数优化路径变得曲折**（zig-zag path），拖慢梯度下降的收敛速度。