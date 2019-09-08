1. Download the image features(folder name - images) from *** and place it in this directory(same as readme)
2. Download the text indices(folder name - text_c10) from *** and place it in this directory
3. Download the glove vectors for dim 50(file name - glove.6B.50d.txt) from *** and place it in this directory



To Run the training, to the folder code in the command prompt and then,
- attribute with MSE loss:
 python att_mse.py
- description with MSE loss:
 python desc_mse.py
- fusion with MSE loss:
 python fusion_mse.py
- fusion with cosine loss:
 python fusion_cosine.py
- fusion with mahalanobis loss:
 python fusion_mah.py

Requirements:
- numpy
- pytorch
- torchfile

Reference:
[1] S. Reed, Z. Akata, B. Schiele, and H. Lee. Learning deep
representations of fine-grained visual descriptions. 
[2] L. Zhang, T. Xiang, and S. Gong, “Learning a deep embedding model for zero-shot learning.
