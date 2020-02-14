cd data
python create_coradatafile.py
cd ../src
python train.py --input ../data/cora.edges --learning-rate 0.1 --num-step 100000 --graph-size 2708 --output ../results/cora.emb
cd ..

python create_data.py
python LightGBM_class.py cora.feature
