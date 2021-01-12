# DeepLearning_COVID19
PyTorch implementation of various state-of-the-art supervised deep learning models for the task of multi-class classification of COVID-19 from chest X-ray imaging. 

Architectures employed:
- ResNet 18 and 50
- VGG16 (batch norm)
- DenseNet-121

This repo complements my other COVID-19 repo, SGAN-COVID19, by generating the results for the supervised architectures. Since the SGAN discriminator utilised the ResNet-18 architecture, but trained it in a semi-supervised way, results with the vanilla supervised model could be produced to compare the learning regimes.

Just like in the SGAN-COVID19 repo a Jupyter Notebook, compatible with Google Colab, has been provided to allow replication of the results. See Experiments in the README.md of SGAN-COVID19 (https://github.com/calum-r-maclellan/SGAN-COVID19/blob/main/README.md).

TODO: add model weights.
