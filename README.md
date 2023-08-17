# Neural-Network
Building a feed neural network from scratch using c++.

## Compilation rules
```
mkdir build
cd build
cmake ..
make
./main [-args] 
```

## Possible arguments

- dataset : "xo" for classification or "y~x" for regression (string)
- biases : Bias Values (double)
- epochs : Number of epochs (int)
- learning rate : Learning rate coefficient (double)
- mu : Nesterov coefficient (double)"
- hidden layer : Number of Nodes of each hidden layer (int ... int)

## Execution examples

```
./main y~x 0 150 0.0001 0.95 2 2
./main xo 0 50 0.003 0.95 8 8 6
```