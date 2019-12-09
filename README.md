# **Deep Learning | Image Binary Classification Model using Intel OpenVino Toolkit**
This repository is part of my final Bachelor´s Degree project. The idea is about take some images with damaged or not areas and identify which one of these owns each category (damaged,undamaged).

![kitten](https://raw.githubusercontent.com/adri1197/DP_Image-Binary-Classification/master/damaged/post_064_056.png "Damaged")
![kitten](https://raw.githubusercontent.com/adri1197/DP_Image-Binary-Classification/master/damaged/post_061_096.png "Damaged")
![kitten](https://raw.githubusercontent.com/adri1197/DP_Image-Binary-Classification/master/undamaged/post_006_127.png "Undamaged")
![kitten](https://raw.githubusercontent.com/adri1197/DP_Image-Binary-Classification/master/undamaged/post_007_094.png "Undamaged")

The main goal is to implement this neural network in Intel´s devices. Hence, we will use the [OpenVino Toolkit](https://software.intel.com/en-us/openvino-toolkit) to obtain a multiplatform and optimize model to be used in Intel´s hardware.
## **Neural Network**
The NN is deployed in Keras  ([Tensorflow])(https://www.tensorflow.org/api_docs/python/tf) using Python. The version of Tensorflow used is **2.0** for development, but for OpenVino is needed **1.15**.
- Conv2D
- 
### **Software**
- I´m using [Google Colab](https://colab.research.google.com/) as my development environment.
- Local Environment
    - OS: Windows 10 Home
    - Python 3.7.4 64-bits
    - OpenVino 2019 R3.1
- Remote Environment
    - OS: Ubuntu 18.04
    - Python 3.6.8
    - OpenVino 2019 R3
### **Hardware**
- Local Environment
    - CPU: Intel i7-6700HQ @2.60GHz - 4 Cores
    - GPU: Intel HD Graphics 530 | NVIDIA GTX950M
    - RAM: 12GB DDR4
- Remote Environment
## **LICENSE**
