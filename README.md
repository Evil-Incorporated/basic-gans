# Basic-GANs
This is the code for our college project (Understanding GANs)

The code for the toy examples can be found [here](https://gist.github.com/Evil-Incorporated/e3040e3d8e1497e0113a46d3a86062c6#file-gan_1d-py):

# MNIST
Our code for training GANs on the MNIST dataset is a minor modification of [pytorch-generative-model-collections](https://github.com/Evil-Incorporated/pytorch-generative-model-collections) by the author [znxlwm](https://github.com/znxlwm)

# LSUN
We used a DCGAN to train on the LSUN outdoor church (as it was the smallest LSUN dataset), the architecture we implemented is exactly the same as mentioned in this paper [Unsupervised representation learning with deep convolutional generative adversarial networks](https://arxiv.org/pdf/1511.06434.pdf). It also contains the WGAN implementation for the church dataset that uses the DCGAN architecture

# Cycle GAN
We used the code from [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN) to train our Cycle GAN on the monet2photo dataset, our modified Cycle GAN is also an almost idententical implementation of this with minor modifications to to implement the fused auto-encoders.  

# Fused encoder Cycle GAN

We modified the Cycle GAN to fuse the encoders to achive faster convergence at the cost of quality picture generation. Check the 
[Reports](https://github.com/Evil-Incorporated/basic-gans/tree/master/Reports) folder to see the results.
