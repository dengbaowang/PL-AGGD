The implemention of PL-AGGD (Adaptive graph guided disambiguation for partial label learning).
MATLAB Commands:
```
ker  = 'rbf'; %Type of kernel function
k = 10;
lambda = 1;
mu = 1;
gama = 0.05;
Maxiter = 10;
run_Lost;
```
Note: Owing to my negligence, the description of the bias b of the regression function and its updating was missed in the paper. 由于我的疏忽，以为实验中用的 multi-regression 部分代码没有偏置项b（其实是有的），在论文中漏掉了对b更新的描述，特此说明，非常抱歉。
