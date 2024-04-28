# CIFAR-100-Image-Classification
The project entails different approaches for the image classification problem, starting with basic CNN and KNN to State of the Art technologies like ViT.

Dataset used - Cifar-100

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

The Vision Transformer (ViT) model implemented in this notebook follows the architecture proposed in the original paper "An Image is Worth 16x16 Words: Transformers for Image Recognition" by Dosovitskiy et al. The model consists of several key components:

The input images are first passed through a data augmentation pipeline, which includes resizing, random horizontal flipping, rotation, and zooming. This preprocessing step enhances the model's ability to generalize to unseen data and improves robustness.

Patch Extraction: The augmented images are divided into patches of a fixed size, typically 16x16 pixels. These patches are then linearized and flattened into vectors, which serve as the input tokens for the Transformer model.

Patch Encoding: Each patch vector is passed through a learnable linear projection layer, which embeds the spatial information of the patch into a high-dimensional feature space. Additionally, a positional embedding is added to each patch to provide spatial context to the model.

Transformer Encoder: The core of the ViT model consists of a stack of Transformer encoder layers. Each encoder layer comprises several sub-layers:
- Multi-Head Self-Attention: This layer allows the model to attend to different parts of the input sequence simultaneously. Each token attends to all other tokens, enabling the model to capture global dependencies within the image.
- Feed-Forward Neural Network: After self-attention, the token representations are passed through a position-wise fully connected feed-forward neural network (MLP). This MLP consists of two linear layers with a GELU activation function in between, which introduces non-linearity to the model.
- Skip Connections and Layer Normalization: Skip connections are employed around each sub-layer, followed by layer normalization. These mechanisms help stabilize training and improve gradient flow, allowing for deeper architectures.

Classification Head: The final token representations after the Transformer encoder layers are passed through a layer normalization and a linear projection layer to obtain the class logits. These logits are then used to compute the probability distribution over the output classes using a softmax activation function.

Overall, the ViT model leverages the self-attention mechanism of Transformers to capture long-range dependencies in images without relying on handcrafted features. By treating images as sequences of patches and applying Transformer-based architectures, ViT achieves competitive performance on various image classification tasks.

epochs = 50
Test Accuracy = 52.63 %
