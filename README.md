# Deep Learning for Music Generation
I love music and deep learning. Let's combine both.

---
**List of experiments**
1. [Overfitting example with Keras](01_Overfitting_Simple_Song.ipynb): Overfit a single and simple piece of music with a basic encoder-decoder LSTM
2. [Overfitting example with PyTorch and Embeddings](https://nbviewer.jupyter.org/github/sorgmi/deeplearning_music_generation/blob/master/02_Overfitting_PyTorch_Embeddings.ipynb) (or use [this link](02_Overfitting_PyTorch_Embeddings.ipynb)): Overfit a single and simple piece of music with a basic encoder-decoder LSTM. Supports different note lengths



**Todo**
- [x] Support variable note length (ties wihtin the same note)
- [ ] Support Ties
- [ ] Support chords
- [ ] Use and analyse attention
- [x] ~~Perform quantization~~
- [ ] Look for more advanced network architectures (RNN's, GAN's...)


## Encoding details
Encoding is inspired by [Bachbot](https://github.com/feynmanliang/bachbot). Image is from [here](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/156_Paper.pdf): 
![Encoding from Bachbot](images/bachbot_encoding.PNG)

Encoding and decoding is done by [encoding.py](pytorchmodels/encoding.py).
Current encoding: **258 different symbols are used as input**
- 128 midi notes: Represent standard notes
- additional 128 midi notes: Represents the same note as the 128 first notes (same pitch), but this time the note is tied to the previous note. This the network can output notes with different lengths (multipple times the same note tied to the previous note).
- 2 additional symbols: START, STOP



&nbsp;
---
**Some interesting and relevant links**
* **http://www.mlsalt.eng.cam.ac.uk/foswiki/pub/Main/ClassOf2016/Feynman_Liang_8224771_assignsubmission_file_LiangFeynmanThesis.pdf**
* https://medium.com/artists-and-machine-intelligence/neural-nets-for-generating-music-f46dffac21c0
* https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/156_Paper.pdf
* https://staff.fnwi.uva.nl/b.bredeweg/pdf/BSc/20152016/Vranken.pdf
* https://www.ini.rub.de/upload/file/1521461530_7126db755dc03bec85b1/dada-bsc.pdf