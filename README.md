# DE-on-taichi
Differential evolution(DE) on taichi and cuda

DE is one kind of evolutionary algorithms that used to solve black-box optimization problem widely.
Examples of ti has not covered evolutionary algorithms which is a field that many researchers concern and interest.
In our code, we prove that the advantage of taichi for evolutionary algorithm which is naturally suitable for parallel computing.

Without strict run time test, DE on taichi is faster than DE on python and matlab because the huge seq loop of iterations. The maximum number of iteration is set to 300,000.

2d res：

![res](https://github.com/Nanase-Nishino/DE-on-taichi/blob/347005dc13c179bad6adb69c707ef13a6b44b160/res.gif)
