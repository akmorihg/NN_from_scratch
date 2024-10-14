
# NN from scratch
## Structure
### Activation Functions
Inherited from abstract parent class **ActivationFunction** and has 2 methods: **forward(x)** and **derivative(x)**.

    1. forward(x) - calculates the function result by given x.
    2. derivative(x) - calculates the derivative value by given x.

Implemented: Relu and Softmax

### Weight Initializers
Inherited from abstract parent class **WeightInitializer** and has **init(n_in, n_out)** method.

    * init(n_in, n_out) - return matrix of size = (n_in, n_out)

Implemented: NormalInitializer, XavierInitializer, PaperInitializer, ZeroInitializer, VeryLargeInitializer

### Optimizer
Inherited from abstract parent class **Optimizer** and has 2 methods: **set_weights_and_bias(weights, bias)** and **optimize(hyperparams, dW, db)**:

    1. set_weights_and_bias(weights, bias) - sets weights to an optimizer.
    2. optimize(hyperparams, dW, db) - changes set weights by passed hyperparams, dW and db.

Implemented: SGD, Momentum, RMSProp, Adam.

### Loss Functions
Inherited from abstract parent class **LossFunction** and has 2 methods: **loss(y_pred, y_true)** and **gradient(y_pred, y_true)**:

    1. loss(y_pred, y_true) - computes loss
    2. gradient(y_pred, y_true) - finds gradient by y_pred

Implemented: MSE, CategoricalCrossEntropy

### Layer
This is class representing layer in NN.

Constructor:
- Layer(n_in, n_out, is_output, weight_init, activation, optimizer):
    - n_in - number of input neurons
    - n_out - number of output neurons
    - weight_init - weight initializer (e.g. XavierInitializer)
    - activation - activation function name
    - optimizer - optimizer name

Methods:

    1. forward(x, is_training) - muntiplies the input by the internal weights and passes through activation function.
    2. backprop(dZ_prev, m, is_batch_mode) - taked a previous loss and evaluates backprop for this layer and returns loss.
    3. update(hyperparams, dW, db) - uses optimizer to update weights and bias.

### NN
Class which stacks provided Layer instances and can compute forward and backprop.

## Experiments
There was a few experiments. For all experiment hyperparams were: learning_rate = 0.1, beta1 = 0.9, beta2 = 0.999, and iterrations = 50 000.

**Blue - Training Loss**

**Oragne - Validation Loss**

### Results

#### NormalInitializer + SGD
Loss:

![loss_img](https://github.com/akmorihg/NN_from_scratch/blob/master/normal_weight_experiment/loss.png?raw=true)

Accuracy:

![accuracy_img](https://github.com/akmorihg/NN_from_scratch/blob/master/normal_weight_experiment/results.png?raw=true)

#### VeryLargeInitializer + SGD:
Loss:

![loss_img](https://github.com/akmorihg/NN_from_scratch/blob/master/large_weight_experiment/loss.png?raw=true)

Accuracy:

![accuracy_img](https://github.com/akmorihg/NN_from_scratch/blob/master/large_weight_experiment/results.png?raw=true)

#### ZeroInitializer + SGD
Loss:

![loss_img](https://github.com/akmorihg/NN_from_scratch/blob/master/zero_weight_experiment/loss.png?raw=true)

Accuracy:

![accuracy_img](https://github.com/akmorihg/NN_from_scratch/blob/master/zero_weight_experiment/results.png?raw=true)

#### XavierInitializer + SGD
Loss:

![loss_img](https://github.com/akmorihg/NN_from_scratch/blob/master/xavier_weight_experiment/loss.png?raw=true)

Accuracy:

![accuracy_img](https://github.com/akmorihg/NN_from_scratch/blob/master/xavier_weight_experiment/results.png?raw=true)

#### PaperInitializer + SGD
Loss:

![loss_img](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weight_experiment/loss.png?raw=true)

Accuracy:

![accuracy_img](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weight_experiment/results.png?raw=true)

#### PaperInitializer + Momentum
Loss:

![loss_img](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_momentum_optimizer_experiment/loss.png?raw=true)

Accuracy:

![accuracy_img](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_momentum_optimizer_experiment/results.png?raw=true)

#### PaperInitializer + RMSProp
Loss:

![loss_img](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_rmsprop_optimizer_experiment/loss.png?raw=true)

Accuracy:

![accuracy_img](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_rmsprop_optimizer_experiment/results.png?raw=true)

#### PaperInitializer + Adam
Loss:

![loss_img](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_adam_optimizer_experiment/loss.png?raw=true)

Accuracy:

![accuracy_img](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_adam_optimizer_experiment/results.png?raw=true)

#### PaperInitializer + Adam, learning_rate = 0.001
Loss:

![loss_img](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_adam_optimizer_proper_lr_experiment/loss.png?raw=true)

Accuracy:

![accuracy_img](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_adam_optimizer_proper_lr_experiment/results.png?raw=true)


### Discussion
The crutial moment which can be highlighted from the experiments is importance of learning_rate. Different Optimizers can do their best if proper learning_rate was choosen. This can be viewed from 'PaperInitializer + Adam' and 'PaperInitializer + Adam, learning_rate = 0.001' experiments.

Also the crutial moment is choosing weights initializer. In the cases of ZeroInitializer, VeryLargeInitializer and NormalInitializer, NN didn't learn anything.

If we compare top results: 'PaperInitializer + SGD' and 'PaperInitializer + Momentum', we can see that 'PaperInitializer + SGD' done better rather than 'PaperInitializer + Momentum', but it is expected 'PaperInitializer + Momentum' be better. This probably can be explained by random nature of weight initialization.

And more interesting results. If we look at weights of the first layer, we can see this:

**PaperInitializer + Momentum weights:**

![weight](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_momentum_optimizer_experiment/weights/weight_1.png?raw=true)
![weight](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_momentum_optimizer_experiment/weights/weight_2.png?raw=true)
![weight](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_momentum_optimizer_experiment/weights/weight_3.png?raw=true)

**PaperInitializer + Adam:**

![weight](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_adam_optimizer_experiment/weights/weight_1.png?raw=true)
![weight](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_adam_optimizer_experiment/weights/weight_2.png?raw=true)
![weight](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_adam_optimizer_experiment/weights/weight_3.png?raw=true)

**PaperInitializer + Adam, learning_rate = 0.001:**

![weight](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_adam_optimizer_proper_lr_experiment/weights/weight_1.png?raw=true)
![weight](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_adam_optimizer_proper_lr_experiment/weights/weight_2.png?raw=true)
![weight](https://github.com/akmorihg/NN_from_scratch/blob/master/paper_weights_adam_optimizer_proper_lr_experiment/weights/weight_3.png?raw=true)

As we see, 'PaperInitializer + Momentum weights' weights are meaningless, but 'PaperInitializer + Adam' weights are seems to have some meaning (or it is an artifact of learning process). When we choose proper learning_rate, again, the weights mostly lost their 'meaning'.
