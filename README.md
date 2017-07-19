# Dog-Cat Classifier
### By Arda Mavi

Dog and cat image classifier with deep learning.<br/>

#### Example:
| <img src="test_dog.jpg?raw=true" width="200">|<img src="test_cat.jpg?raw=true" width="200">|
|:-:|:-:|
|Dog: 0.92035621<br/>Cat: 0.04618423|Cat: 0.90135497<br/>Dog: 0.09642436|

##### Layer outputs of test photographs:

| <img src="Data/Layer_Outputs/Dog/Layer_1_Outputs/4.jpg?raw=true" width="32"> <img src="Data/Layer_Outputs/Cat/Layer_1_Outputs/4.jpg?raw=true" width="32">| <img src="Data/Layer_Outputs/Dog/Layer_2_Outputs/16.jpg?raw=true" width="32"> <img src="Data/Layer_Outputs/Cat/Layer_2_Outputs/16.jpg?raw=true" width="32">| <img src="Data/Layer_Outputs/Dog/Layer_3_Outputs/10.jpg?raw=true" width="32"> <img src="Data/Layer_Outputs/Cat/Layer_3_Outputs/10.jpg?raw=true" width="32">|
|:--:|:--:|:--:|
|Layer: 1<br/>Kernel: 4|Layer: 2<br/>Kernel: 16|Layer: 3<br/>Kernel: 10|

###### Look up `Data/Layer_Outputs` folder for other outputs.

### Using Predict Command:
`python3 predict.py <ImageFileName>`

### Model Training:
`python3 train.py`

### Using TensorBoard:
`tensorboard --logdir=Data/Checkpoints/./logs`

### Model Architecture:
- Input Data
Shape: 64x64x3

Layer 1:
- Convolutional Layer
32 filter
Filter shape: 3x3

- Activation
Function: ReLu

- Max Pooling
Pool shape: 2x2

Layer 2:
- Convolutional Layer
32 filter
Filter shape: 3x3

- Activation
Function: ReLu

- Max Pooling
Pool shape: 2x2

Layer 3:
- Convolutional Layer
64 filter
Filter shape: 3x3

- Activation
Function: ReLu

- Max Pooling
Pool shape: 2x2

Classification:
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

If you want to add a new training image to previously category datasets, you add a image to about category directory and if you have `npy` files in `Data` folder delete `npy_train_data` folder.

Note: We work on 64x64 image also if you use bigger or smaller, program will automatically return to 64x64.

### Important Notes:
- Used Python Version: 3.6.0
- Install necessary modules with `sudo pip3 install -r requirements.txt` command.
