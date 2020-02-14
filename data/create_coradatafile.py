import networkx as nx
import numpy as np
import pickle as p
from matplotlib import pyplot as plt

id = 0
nodes_map = {}
label_file = open('raw/cora/cora.content')
output1 = open("map.file", "w")
for line in label_file:
    nums = line.split()
    nodes_map[nums[0]] = id
    output1.write(nums[0] + " " + str(id) + "\n")
    id += 1
output1.close()


graph_file = open('raw/cora/cora.edges', 'r')
graph_file.seek(0)
cora_edgelist = []
for line in graph_file.readlines():
    i, j = line.split()
    index_i, index_j = nodes_map[i], nodes_map[j]
    cora_edgelist.append((index_i, index_j))
cora = nx.DiGraph(cora_edgelist)

'''
# Get a conversion dictionary
lookup = {}
for new_ids, ids in enumerate(cora.nodes()):
    lookup[ids] = new_ids
# Create new graph with new node ids
new_cora = nx.Graph()
for i, j in cora.edges():
    new_cora.add_edge(lookup[i], lookup[j])
print(len(list(new_cora.nodes())))
'''
'''
cora_labels = np.ndarray(shape=len(new_cora), dtype=int)
cora_features = np.ndarray(shape=(len(new_cora), 1433), dtype=int)
content = 'raw/cora/cora.content'
labels = {'Case_Based': 0, 'Genetic_Algorithms': 1, 'Neural_Networks': 2, 
          'Probabilistic_Methods': 3, 'Reinforcement_Learning':4, 
          'Rule_Learning': 5, 'Theory': 6}
with open(content, 'r') as f:
    for lines in f.readlines():
        idx, *data, label = lines.strip().split()
        idx = int(idx)
        cora_labels[lookup[idx]] = labels[label]
        for i, val in enumerate(map(int, data)):
            cora_features[lookup[idx]][i] = val

with open(content, 'r') as f:
    for lines in f.readlines():
        idx, *data, label = lines.strip().split()
        idx = int(idx)
        assert sum(cora_features[lookup[idx]]) == sum(map(int, data))
        assert cora_labels[lookup[idx]] == labels[label]
    print("Sanity test for cora_features and cora_labels passed.")            
    
from scipy.sparse import csr_matrix
cora_csr_features = csr_matrix(cora_features)
cora_dataset = {'NXGraph': new_cora, 'Labels': cora_labels, 'CSRFeatures': cora_csr_features}
with open("./../data/cora.data", 'wb') as f:
    p.dump(cora_dataset, f, protocol=2)   
'''    

cora_dataset = {'NXGraph': cora}
with open("cora.data", 'wb') as f:
    p.dump(cora_dataset, f, protocol=2)
