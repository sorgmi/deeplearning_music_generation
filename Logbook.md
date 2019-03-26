### 26. März 2019 16:06
- Optimizer has a huge influence!
- Implemented basic encoder-decoder with pyorch. With SGD I was not able to overfit  a single song (even after 2000 epochs and huge models). With ADAM is was no problem at all (100 epochs were fine)!
- Overfitting a single song with SGD using categorical labels + Softmax was no problem. But multi-label + Sigmoid fails completely with SGD