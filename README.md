# Movie-Genre-Classification

Using VGG16 - Convolutional Network for movie genre classification on the basis of movie poster.

VGG16 Architecture:The input to cov1 layer is of fixed size 224 x 224 RGB image. The image is passed through a stack of convolutional (conv.) layers, where the filters were used with a very small receptive field: 3×3 (which is the smallest size to capture the notion of left/right, up/down, center). In one of the configurations, it also utilizes 1×1 convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity). The convolution stride is fixed to 1 pixel; the spatial padding of conv. layer input is such that the spatial resolution is preserved after convolution, i.e. the padding is 1-pixel for 3×3 conv. layers. Spatial pooling is carried out by five max-pooling layers, which follow some of the conv.  layers (not all the conv. layers are followed by max-pooling). Max-pooling is performed over a 2×2 pixel window, with stride 2.

![alt text](https://neurohive.io/wp-content/uploads/2018/11/vgg16-neural-network.jpg)

The code also consists of poster fetching and then recommending similar movie names and movie links to the user.Enter the nameof the movie and voila it will tell the genre of the movie with the poster and will recommend 5 movies to the user.


# Flask Web App
![screenshot](Capture.png)
