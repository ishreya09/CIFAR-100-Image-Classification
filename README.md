# CIFAR-100-Image-Classification
The project entails different approaches for the image classification problem, starting with basic CNN and KNN to State of the Art technologies

## Models Used

#### CNN + KNN

For CNN, convolution layers are established to extract local features followed by a max pooling layer. Max pooling layer helps us out in capturing the major features, decreasing the model complexity. It also helps us in translational invariance (position of object in the image will not matter). After repeating this set, I have introduced FCNN with Dense layers with output layer having softmax activation. The loss function utilized is Sparse Categorical Cross Entropy. The optimizer used is Adam. The metric utilized is accuracy. The second last layer (Dense layer) output is then utilized to send into a KNN Classifier model and then trained again.

epochs = 10
Test Accuracy: 30.23

#### Resnet50 + FCNN

ResNet50 architecture contains 48 convolution layer, 1 max pool layer and 1 average pooling layer. The architecture includes : 
- 7x7 kernel conv with 64 filters with stride=2
- Multiple layers with different kernel size and filter counts(1x1,3x3 and 1x1)
- Bottleneck blocks with 1x1, 3x3 and 1x1 convolutions.
- Iterations of above layers complete ResNet50.

Features: 
- Skip connections: ResNet50 introduces shortcut connections that skip over some layers. These connections allow gradients to flow directly through the network, adressing the vanishing gradient problem. This converts a regular network to residual network
- Bottleneck Design: A bottleneck design block includes 1x1 convolutions(known as bottleneck) to reduce the number of parameters and matrix multiplication.
- Stacked layers: Multiple layers are stacked. It includes stack of 3 layers instead of 2 which enhances expressiveness. 

epochs=10
Test Accuracy: 21.79

#### ViT



epochs = 50
Test Accuracy = 
