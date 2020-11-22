# NVAE-TF
![](sample.png)

A TensorFlow implementation of NVAE.

Details of our implementation and a discussion of the results is available in
the PDF `DD2412_Final_Project_NVAE.pdf`.

The code in this repository is currently quite disorganized.
The main file is `train.py` which can be executed with the
flag `python train.py -h` in order to find out about the various
options that can be provided as command line flags. The defaults 
are set to the hyperparameters that were suggested in the source
paper for MNIST. The model itself is split between the files `models.py`
which contains the main NVAE class, and the four files `encoder.py`,
`decoder.py`, `preprocess.py` and `postprocess.py` which contain 
classes forming the four components of the NVAE architecture.

Some functionality currently resigns in other branches. Specifically,
an implementation that uses spectral regularization instead of 
spectral normalization is available on the branch `spectral_reg`, and 
the other branches are related to various tweaks to the evaluation
metrics.

Extending the code to other datasets should not be more difficult
than replacing the use of Bernoulli distributions with mixed Gaussians
and writing a new dataset loader.

## Results
| Model                  | NLL (nats)     | FID           | Precision          | Recall             | Training time (h) |
| ---------------------- | -------------- | ------------- | ------------------ | ------------------ | ----------------- |
| *Ours*                 |                |               |                    |                    |                   |
| Step + SN              | 87.06 (+-2.18) | 8.87          | 0.8950 (+- 0.0999) | 0.9227 (+- 0.0879) | 49                |
| Step + SR              | 80.33 (+-2.01) | 30.37         | 0.8559 (+- 0.0608) | 0.8803 (+- 0.0546) | 104               |
| Epoch + SN             | 98.92 (+-1.83) | 20.85         | 0.7541 (+- 0.152)  | 0.8828 (+- 0.114)  | 71                |
| *Others*               |                |               |                    |                    |                   |
| Vanilla VAE            | 86.76          | 28.2 (+- 0.3) | -                  | -                  | -                 |
| NVIDIA's NVAE w/o flow | 78.01          | -             | -                  | -                  | -                 |
| NVIDIA's NVAE w/ flow  | 78.19          | -             | -                  | -                  | -                 |
| Generative Latent Flow | -              | 5.8 (+- 0.1)  | 0.981              | 0.963              | -                 |
| PixelCNN               | 81.30          | -             | -                  | -                  | -                 |
| LMCONV                 | 77.58          | -             | -                  | -                  | -                 |
