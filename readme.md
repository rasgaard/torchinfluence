# TorchInfluence

Implementation of Training Data Attribution (TDA) methods using PyTorch - namely [torch.func](https://pytorch.org/docs/stable/func.html).

TDA methods attempt to attribute a score to training points in relation to how important or influential they are for the prediction of a given test point.

For now I have only implemented the simple gradient similarity as proposed in \[4\] which also is a key element of TracIn \[2\] as a warmup exercise but I plan on working towards an implementation of influence functions that utilize Arnoldi iterations to efficiently estimate the inverse Hessian as done in \[3\].


## Literature

\[1\] Koh, P. W., & Liang, P. (2017, July). [Understanding black-box predictions via influence functions.](http://proceedings.mlr.press/v70/koh17a) In International conference on machine learning (pp. 1885-1894). PMLR. <br>
\[2\] Pruthi, G., Liu, F., Kale, S., & Sundararajan, M. (2020). [Estimating training data influence by tracing gradient descent.](https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html) Advances in Neural Information Processing Systems, 33, 19920-19930. <br>
\[3\] Schioppa, A., Zablotskaia, P., Vilar, D., & Sokolov, A. (2022, June). [Scaling up influence functions.](https://ojs.aaai.org/index.php/AAAI/article/view/20791) In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 8, pp. 8179-8186). <br>
\[4\] Charpiat, G., Girard, N., Felardos, L., & Tarabalka, Y. (2019). [Input similarity from the neural network perspective.](https://proceedings.neurips.cc/paper/2019/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html) Advances in Neural Information Processing Systems, 32.