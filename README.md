# pattern-generation

## Kernel function
A kernel is essentially a fixed size array of numerical coefficients along with an anchor point in that array, which is typically located at the center.

In order to be a valid kernel function the resulting kernel matrix should be positive definite . Which implies that the matrix should be symmetric . Being positive definite also means that the kernel matrix is invertible.

### Exponentiated quadratic kernel
![wq1](https://github.com/ashleetiw/pattern-generation/blob/master/eq1.png)

![eq1](https://github.com/ashleetiw/pattern-generation/blob/master/f1.png)

### Rational quadratic kernel
![wq2](https://github.com/ashleetiw/pattern-generation/blob/master/eq2.png)

![eq2](https://github.com/ashleetiw/pattern-generation/blob/master/f2.png)

## Periodic kernel
![wq3](https://github.com/ashleetiw/pattern-generation/blob/master/eq3.png)

![eq3](https://github.com/ashleetiw/pattern-generation/blob/master/f3.png)


<!-- ## Heat Map and activation Map for analysis 
![a1](../assets/img/activ1.png){: width="200" }
![a2](../assets/img/act3.png){: width="300" }
![a3](../assets/img/act2.png){: width="350" } -->

## References
1:[https://www.cs.toronto.edu/~duvenaud/cookbook/](https://www.cs.toronto.edu/~duvenaud/cookbook/)
2:[http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)