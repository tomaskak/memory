# Memory
An experimental package that attempts to create a form of memory (whether an input has been seen or not) by compressing successive inputs 
into a latent form via an autoencoder. The goal is for the memory encoder to be used as a fixed structure in another architecture that would 
benefit from an RNN such as an RL agent or an NN with a sequence as input.

## Diagram
```
M_t - Memory at time t
X_t - Input at time t
X_q - Input for querying
E - Encoder
D - Decoder

M_t --->  E ---> M_t+1 ---> D ---> (Y/N)
X_t _____/       X_q ______/

```
In this diagram the encoder is being trained to take a memory in its latent form and a new observation and to compress these inputs into a 
new memory value. This memory value is then trained by using it to construct a decoder that accepts inputs that should and should not be 
present in the memory, expecting the decoder to learn correctly whether Y the input is in the memory or N it is not.

## Training considerations
### Data
What data is being compressed? 
* Random - This should do as well as there is space in the memory size.
* Monotonic increasing paths - Each input is a step from the previous input thereby creating a path. The encoder would ideally learn to compress by storing the most important points (points with sharp angles).

### Incorrect Values
How to generate the incorrect samples?
* Compare against different samples from the same data source
* Random
* Same as input with added noise to [1-P] dimensions

### Loss
How to compute the loss for correct/incorrect inputs?
* 0/1 loss, no weighting
* Weight correct vs incorrect differently
* Weight incorrect values according to their Euclidean distance to the closest correct input in M
