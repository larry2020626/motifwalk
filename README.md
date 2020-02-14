# MotifWalk: Network local structural representation embedding.

all credits go to Hoang NT https://github.com/gear/motifwalk

This project is an implement of paper Motif-Aware Graph Embeddings

Based on the author's code, I sorted it out and add classifier and preprocess part.

## Environment

MotifWalk is developed using Python 3.5.2. Additional packages:

- NetworkX 1.11
- Tensorflow 0.10.0rc0
- Sklearn \& Scipy 0.18.1
- Numpy 1.11.2


##### How to run
```
cd data
python create_coradatafile.py
cd ../src
python train.py --input ../data/cora.edges --learning-rate 0.1 --num-step 100000 --graph-size 2708 --output ../results/cora.emb
cd ..

python create_data.py
python LightGBM_class.py cora.feature

```