
# Introduction

## Intiuition and math:

If we consider a classification problem with the following data:

![Picture 1: Example data for classification.](https://docs.microsoft.com/en-us/cognitive-toolkit/tutorial/synth_data.png "Example data")

We have two input dimensions `x1` and `x2`, as well as output labels `y`. Our job is to classify a given input of arbitrary values `x1` and `x2`, and assign them to class `0` or `1`.

![Picture 2: Neural Network representation.](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/1200px-Colored_neural_network.svg.png "Neural Network representation.")

Let's take a look a the following neural network (NN):

$$PARAMS = [X, Y, W_1, b_1, W_2, b_2] \\ Z_1 = W_1*X + b_1 \\ A_1 = \sigma(Z_1) \\ Z_2 = W_2 * A_1 + b_2 \\ A_2 = \sigma(Z_2) \\ L(A_2, Y) => \hat{y} \\ RETURNS = [Z_1, A_1, Z_2, A_2, L(A_2, Y), \hat{y}]$$

*Note that to simplify the calculations the input now only has one input feature `X`.*

A typical NN consists of multiple layers of neurons that will make decisions, based on provided data and parameters. The parameters $X$ and $Y$ are part of the problem set. $X$ contains multiple input values that help the NN approach the label value $Y$, where the output of the NN is typically referred to as $\hat{y}$. The difference between $Y$ and $\hat{y}$ will determine the loss of the NN. Our goal is to find parameter values for $W_1, b_1, W_2, b_2$ that minimize the loss as much as possible.

In the above NN, $\sigma$ is an activation function to make the output of a layer non-linear (we will use sigmoid), and $L$ is a loss function that satisfies the given problem (in our case cross entropy).

As a side note, these two links should help understand why activation functions are neccessary:

- [https://www.wolframalpha.com/input/?i=.6%28.5*x+%2B+.2%29+%2B+.1](https://www.wolframalpha.com/input/?i=.6%28.5*x+%2B+.2%29+%2B+.1)
- [https://www.wolframalpha.com/input/?i=.6%28sigmoid%28.5*x+%2B+.2%29%29%2B.1](https://www.wolframalpha.com/input/?i=.6%28sigmoid%28.5*x+%2B+.2%29%29%2B.1)

In our initial NN, all of these operations are chained together, meaning that $\hat{y}$ could also be represented as one long equation. An algorithm called Backpropagation can be used to calculate the gradients of all learnable parameters $(W_1, b_1, ...)$. This will give us the contribution of each parameter to the overall loss. Gradient Descent is used to minimize the overall loss to a (local) minimum.

## Backpropagation

![Picture 3: The image shows us how we move from a start position to a local minimum by adjusting our parameters, one step at a time.](https://cdn-images-1.medium.com/max/1600/1*f9a162GhpMbiTVTAua_lLQ.png "Gradient Descent")

To be able to determine how we can walk downhill using Gradient Descent, we have to know what the slope (or gradient) of our current position is.

![Picture 4: The slope of the current function at a given point.](https://sebastianraschka.com/images/faq/closed-form-vs-gd/ball.png "Slope using Gradient Descent")

The simplest way to calculate the slope is by using the formula
$$b = \frac{(x - x`)(y - y`)}{(x - x`)^2}$$, which requires us to have two points on the slope. In our problem we have a multi dimensional landscape, and it is therefore almost impossible to find two points that would help us nagivate "down hill". Calculus thought us that the derivative of a function gives us the slope at a specific point. In other words, we don't have to find multiple points to get the gradient. The complexity of the required derivative increases with the depth of the NN. Luckily, because all the operations of the NN are subfunctions, we can use the chain rule to solve for desired gradients:
$$\frac{dL}{dW} = \frac{dL}{dA}*\frac{dA}{dZ}*\frac{dZ}{dW}.$$

To give this some more context, let's calculate the derivative of the loss function $L$ with respect to $A_2$. We'll use cross entropy as our loss function for this example:
$$\frac{dL}{dA_2}[-\frac{1}{m}\sum^{m}_{i=0}(Y*log(A_2)+(1-Y)*log(1-A_2))] = \color{red}{-\frac{Y}{A_2}+\frac{1-Y}{1-A_2}}.$$ 

Next up we have to calculate the derivative of our activation function. For our example, we'll use the sigmoid function, as it limits the values to a range of [0, 1]:
$$\frac{dA_2}{dZ_2}[\frac{1}{1+e^{-Z_2}}] \\ = \frac{dA}{dZ}[(1+e^{-Z_2})^{-1}] \\ = -(1+e^{-Z_2})^{-2}(-e^{-Z_2}) \\ = \frac{e^{-Z_2}}{(1+e^{-Z_2})^2} \\ = \frac{1}{1+e^{-Z_2}}\frac{1+e^{-Z_2}-1}{1+e^{-Z_2}} \\ = \frac{1}{1+e^{-Z_2}}(1-\frac{1}{1+e^{-Z_2}}) \\ = \color{red}{A_2(1-A_2)}$$
To show the principle of the chain rule, let's calculate $\frac{dL}{dZ_2}$:
$$\frac{dL}{dZ_2} = \frac{dL}{dA_2} * \frac{dA_2}{dZ_2} \\ = (-\frac{Y}{A_2}+\frac{1-Y}{1-A_2})(A_2)(1-A_2) \\ = (-Y+\frac{A_2(1-Y)}{(1-A_2)})(1-A_2) \\ = -Y(1-A_2)+A_2(1-Y) \\ = \color{red}{A_2-Y}.$$

Now we'll start the gradient calculation of our first parameter, $W_2$:
$$\frac{dZ_2}{dW_2}[W_2*A_1 + b_2] = \color{red}{A_1},$$ which means that for the contribution of $W_2$
$$\frac{dL}{dW_2} = \frac{dL}{dZ} * \frac{dZ}{dW_2} =  \color{red}{(A_2-Y)A_1}.$$

Similarly we also have to know $\frac{dL}{db_2}$:
$$\frac{dZ_2}{db_2}[W_2*A_1 + b_2] = \color{red}{1},$$ and again
$$\frac{dL}{db_2} = \frac{dL}{dZ} * \frac{dZ}{db_2} = \color{red}{(A_2-Y)}.$$

In theory these would be all the calculations needed, if we only had one layer in our NN. Because we also have to calculate the gradients for $W_1$ and $b_1$, we need to first calculate the gradients for $A_1$, since they propagate through the same function as $W_2$ and $b_2$. We have already done the required work for this:
$$\frac{dZ_2}{dA_1}[W_2*A_1 + b_2] = \color{red}{W_2},$$ which means that
$$\frac{dL}{dA_1} = \frac{dL}{dZ} * \frac{dZ_2}{dA_1} = \color{red}{(A_2-Y)W_2}.$$

Now we continue in the same fashion as we did for $W_2$ and $b_2$, and will be able to obtain $W_1$ and $b_1$. So,
$$\frac{dL}{dW_1} = \frac{dL}{dA_1} * \frac{dA_1}{dZ_1} * \frac{dZ_1}{dW_1},$$ and
$$\frac{dL}{db_1} = \frac{dL}{dA_1} * \frac{dA_1}{dZ_1} * \frac{dZ_1}{db_1}.$$

## Gradient Descent

For each of the obtained gradients, we now calculate the new weight:
$$W = W - \alpha*\frac{dL}{dW},$$
where alpha is a so called hyper parameter, which can't be learned, but has to be provided at the beginning of training.

## Live example:

View this NN in action: [Tensorflow JS Playground](https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1,1&seed=0.43153&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

