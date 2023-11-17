# TorchInfluence

Implementation of Training Data Attribution (TDA) methods using PyTorch - namely torch.func.

TDA methods attempt to attribute a score to training points in relation to how important or influential they are for the prediction of a given test point.

For now I have only implemented the simple gradient similarity part of TracIn as a warmup exercise but I plan on working towards an implementation of influence functions that utilize Arnoldi iterations to efficiently estimate the inverse Hessian as done in \[3\].


## Literature

\[1\] [Koh, P. W., & Liang, P. (2017, July). Understanding black-box predictions via influence functions. In International conference on machine learning (pp. 1885-1894). PMLR.](http://proceedings.mlr.press/v70/koh17a) <br>
\[2\] [Pruthi, G., Liu, F., Kale, S., & Sundararajan, M. (2020). Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems, 33, 19920-19930.](https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html) <br>
\[3\] [Schioppa, A., Zablotskaia, P., Vilar, D., & Sokolov, A. (2022, June). Scaling up influence functions. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 8, pp. 8179-8186).](https://ojs.aaai.org/index.php/AAAI/article/view/20791)