# DE-on-taichi

this repo has been merged into taichi.examples

Differential evolution(DE) on taichi and cuda

DE is one kind of evolutionary algorithms that used to solve black-box optimization problem widely.
Examples of ti has not covered evolutionary algorithms which is a field that many researchers concern and interest.
In our code, we prove that the advantage of taichi for evolutionary algorithm which is naturally suitable for parallel computing.

Without strict run time test, DE on taichi is faster than DE on python and matlab because the huge seq loop of iterations. The maximum number of iteration is set to 300,000.

2d res：

![res](https://github.com/Nanase-Nishino/DE-on-taichi/blob/ed4d1c6b3b4567b51d3b2681518f889b199d050d/2dres.gif)

3d res:

![res](https://github.com/Nanase-Nishino/DE-on-taichi/blob/46fb2c1c3c4c3ddc26f24c07b8223bb53b8f39b9/3dres.gif)
