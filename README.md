# Hierarchical-Model-for-Long-term-Video-Prediction
Given the previous frames of the video as input, we want to get the long-term frame prediction.

Adopted human pose prediction method based on: Villegas, Ruben, et al. "Learning to Generate Long-term Future via Hierarchical Prediction." arXiv:1704.05831 (2017)

Authors: Peter Wang, Zhongxia Yan, Jeffrey Zhang

Note: 1. The latest version of tensorflow is needed.

2. To run our code, you should first get Penn Action Dataset (Weiyu Zhang, Menglong Zhu and Konstantinos Derpanis, "From Actemes to Action: 
A Strongly-supervised Representation for Detailed Action Understanding" International Conference on Computer Vision (ICCV). Dec 2013.):
http://dreamdragon.github.io/PennAction/

and then run preposs.ipynb

3. Run our LSTM model and analogy network model separately.