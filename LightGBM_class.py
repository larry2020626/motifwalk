import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import lightgbm as lgb

path_train = sys.argv[1] + '_train'
path_test = sys.argv[1] + '_test'
path_val = sys.argv[1] + '_val'

dtrain = lgb.Dataset(path_train)
dtest = lgb.Dataset(path_test)
dval = lgb.Dataset(path_val)
test_l = open(path_test)
labels = []

#load label for test
for line in test_l.readlines():
    nums=line.split()
    labels.append(float(nums[0]))

param = {'learning_rate':0.1, 'num_leaves':31, 'objective':'binary'}
bst = lgb.train(param, dtrain, 2000, valid_sets = dval, early_stopping_rounds = 30)
print(path_test)
preds = bst.predict(path_test)
print(preds)

res = [[0,0],[0,0]]
for i in range(len(preds)):
    if labels[i] == 1:
        if abs(preds[i]) >= 0.5:
            res[0][0] += 1
        else:
            res[0][1] += 1
    if labels[i] == 0:
        if abs(preds[i]) <= 0.5:
            res[1][1] += 1
        else:
            res[1][0] += 1
print(res)
loss = metrics.mean_squared_error(labels,preds)
print(loss)
Acc = 1.0 * (res[0][0] + res[1][1]) / (res[0][0] + res[0][1] + res[1][0] + res[1][1])
precision, recall = 1.0 * res[0][0] / (res[0][0] + res[0][1]), 1.0 * res[0][0] / (res[0][0] + res[1][0])
if precision + recall != 0:
    F1 = 2 * precision * recall / (precision + recall)
else:
    F1 = 0
output = open("logging.file", "a")
print("Accuracy, precision, recall, F1: " + str(Acc) + " " + str(precision) + " " + str(recall) + " " + str(F1) + "\n")
output.write(str(Acc) + " " + str(precision) + " " + str(recall) + " " + str(F1) + "\n")
output.close()
