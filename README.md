# Dog-Cat-Classifier

## Arda Mavi - [ardamavi.com](http://www.ardamavi.com/)

Dog and cat image classifier with deep learning

I use Keras for making deep learning model.<br/>
Keras is a simple and strong deep learning library, written in Python and running with TensorFlow and Theano.<br/>
I use TensorFlow background in Keras because Tensorflow has better multiple GPUs support than Theano.<br/>

### Using Commend: <br/>
`python3 predict.py <ImageFileName>`

### Architecture:
- Input Data
Shape: 64x64x3

- Convolutional Layer
32 filter
Filter shape: 3x3

- Activation
Function: ReLu

- Max Pooling
Pool shape: 2x2

- Convolutional Layer
32 filter
Filter shape: 3x3

- Activation
Function: ReLu

- Max Pooling
Pool shape: 2x2

- Convolutional Layer
64 filter
Filter shape: 3x3

- Activation
Function: ReLu

- Max Pooling
Pool shape: 2x2

- Flatten

- Dense
Size: 64

- Activation
Function: ReLu

- Dropout
Rate: 0.5

- Dense
Size: 2

- Activation
Function: Sigmoid

##### Optimizer: Adadelta
##### Loss: Binary Crossentropy

### Adding new train dataset:
If you want to add new dataset to datasets, you create a directory and rename what you want to add category (like 'cat' or 'phone').

If you want to add a new training image to previously category datasets, you add a image(suggested minimum image size: 64x64) to about category directory and if you have `npy` files in `Data` folder delete `npy_train_data` folder.

Note: We work on 64x64 image also if you use bigger or smaller, program will automatically return to 64x64.

### Used Modules:
- tensorflow
- scikit-learn
- scikit-image
- numpy
- keras
- h5py

### Important Notes:
- Install above modules
