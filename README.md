# PyNN

Simple Python3 Recurrent Neural Networks framework.

## Structure and State
In the PyNN structure and state are divided. Structure described by `Nodes`, `Paths` and `Networks`. State is represented by `_State` for forward propagation, `_Error` for backward error propagation, and `_Gradient` for storing computed gradient.

## Nodes
There are some types of nodes:
+ `MatrixProduct`
+ `Bias`
+ `Tanh`
+ `Rectifier`
+ `Mixer`
    + `Fork`
    + `Join`

## TODO
- [x] Separate structure and state model
- [x] AdaGrad
- [ ] Loss layers
- [ ] Pre-defined common networks
- [ ] Multithreading
- [ ] GPU Acceleration
