# MNIST seq-seq models

The MNIST sequential problem could be phrased in two ways:
1.  Row by Row, where we pass one row at a time to our model, totaling 28 sequences.
2.  Pixel by Pixel, where we pass on the pixel at a time, totaling 28*28 sequences.


## Implemntation Details:

### Shared Hyper Parameters:

* CrossEntropyLoss 
* SGD with Learning Rate = 0.01
* Batch size= 100
* Number of Epochs= 75
* Validation Percentage = 10



### RNN 

1. 3 total layers
2. Each hidden layer has 128 neurons
3. The RNN was defined with an input size of 28 (sequence length). We also set the batch first to true since our input data shape is the following (batch size, sequence length, h, w).
4. A fully connected layer with a size of (128,10) where 10 is the number of classes.

##### Forward Method:

1. Initialize the hidden state with zeros.
2. The hidden shape is the following (3, 28, 128).
3. This is followed by an RNN pass, this returns the output of the following shape  (batch size, 28, hidden size)
4. Index the last time step, which is then passed to a Fully Connected Layer, which in turns return the following shape (batch size, 10) 
5. Softmax activation has been applied to the output.

### LSTM

1. 3 total layers
2. Each hidden layer has 128 neurons
3. The RNN was defined with an input size of 28 (sequence length). We also set the batch first to true since our input data shape is the following (batch size, sequence length, h, w).
4. A fully connected layer with a size of (128,10) where 10 is the number of classes.

##### Forward Method:

1. Initialize the hidden state with zeros.
2. Initialize the cell state with zeros. 
3. The hidden shape is the following (3, 28, 128).
4. This is followed by an LSTM  pass, this returns the output of the following shape  (batch size, 28, hidden size)
5. Index the last time step, which is then passed to a Fully Connected Layer, which in turns return the following shape (batch size, 10) 
6. Softmax activation has been applied to the output.


### GRU

1. 3 total layers
2. Each hidden layer has 128 neurons
3. The RNN was defined with an input size of 28 (sequence length). We also set the batch first to true since our input data shape is the following (batch size, sequence length, h, w).
4. A fully connected layer with a size of (128,10) where 10 is the number of classes.
5. ReLU activation.

##### Forward Method:

1. Initialize the hidden state with zeros.
2. The hidden shape is the following (3, 28, 128).
3. This is followed by an RNN pass, this returns the output of the following shape  (batch size, 28, hidden size)
4. We then Index the last time step, which is then passed to the Fully Connected layer after the ReLU is applied. which in turns return the following shape (batch size, 10) 
5. Softmax activation has been applied to the output.


### Transformer Model

The Architecture is presented in the following paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. 


1. The input image is split into fixed-size patches.
2. The positional encoder is the number of the patch in the input image.
3. The embedding is passed into a TransfomerEncoder (multiple multi-headed self-attention layers).
4. A learnable classification token has been added to the sequence to perform the classification.

The input 2D images are reshaped into compressed 2D patches to change the image data to match the transformer input. The series of embedded patches were prepended with a learnable embedding. This token has the same function as BERT's [CLS] token. The fully connected MLP head at the output provides the desired class prediction.

Https://github.com/lucidrains/vit-pytorch implemented the code. 

We have used the ViT model with the following parameters:

1. Patch size of 6*6.
2. Embedding dimension of 64.
3. A depth of 6 Transformer blocks.
4. 8 transformer heads.
5. 128 units in the hidden layer of the output MLP head.
  
