import pickle as p
import numpy as np
import re
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from time import time
try:
    from graph_tool.clustering import motifs, motif_significance
    from graph_tool.spectral import adjacency
    from graph_tool import load_graph_from_csv
except ImportError:
    print("Warning: graph_tool module is missing, motif analysis is not available.")

dataloc = './../data/'


def load_citeseer():
    with open(dataloc+'citeseer.data', 'rb') as f:
        data = p.load(f)
        graph = data['NXGraph']
        features = data['CSRFeatures']
        labels = data['Labels']  # Number format
        labels = MultiLabelBinarizer().fit_transform(
                        labels.reshape(labels.shape[0], 1))
    return graph, features, labels


def load_cora():
    with open(dataloc+'cora.data', 'rb') as f:
        data = p.load(f)
        graph = data['NXGraph']
        features = data['CSRFeatures']
        labels = data['Labels']  # Number format
        labels = MultiLabelBinarizer().fit_transform(
                        labels.reshape(labels.shape[0], 1))
    return graph, features, labels


def load_blogcatalog():
    with open(dataloc+'blogcatalog.data', 'rb') as f:
        data = p.load(f)
        graph = data['NXGraph']
        features = None
        labels = data['LILLabels']  # Already in binary format
    return graph, features, labels


def load_data(dataset_name):
    """Load dataset"""
    if dataset_name == "blogcatalog":
        return load_blogcatalog()
    elif dataset_name == "cora":
        return load_cora()
    elif dataset_name == "citeseer":
        return load_citeseer()
    else:
        raise ValueError("Dataset not found")


def load_embeddings(emb_file):
    """Load graph embedding output from deepwalk, n2v to a numpy matrix."""
    with open(emb_file, 'rb') as efile:
        num_node, dim = map(int, efile.readline().split())
        emb_matrix = np.ndarray(shape=(num_node, dim), dtype=np.float32)
        for data in efile.readlines():
            node_id, *vector = data.split()
            node_id = int(node_id)
            emb_matrix[node_id, :] = np.array([i for i in map(np.float, vector)])
    return emb_matrix


def write_motifs_results(output, motifs_list, z_scores,
                         n_shuf, model="uncorrelated"):
    """Write the adjacency matrix of motifs in the `motifs_list`
    and its corresponding z-scores to file.
    Naming convention: blogcatalog_3um.motifslog
    Parameters
    ==========
        output: name of file will be writen to disk
        motifs_list: list of motifs (graph-tool graph instances)
        z_scores: corresponding z_scores to each motif.
        n_shuf: number of random graph generated
        model: name of the random graph generation algorithm"""
    assert len(motifs_list) == len(z_scores)
    with open(output, 'w') as f:
        f.write("Number of shuffles: {} ({})\n".format(n_shuf, model))
        for i, m in enumerate(motifs_list):
            f.write("Motif {} - z-score: {}\n".format(i+1, z_scores[i]))
            for rows in adjacency(m).toarray():
                f.write(str(rows) + '\n')
            f.write('\n')
    return output


def run_motif_significance(graph, directed=True, data_loc="../data/", motif_size=3,
                           n_shuffles=16, s_model='uncorrelated'):
    """Run z-score computation for all `motif_size` subgraph
    on the given `graph`. By default, graph is loaded as a directed graph_tool
    instance.
    Parameters
    ==========
        graph: name of the graph file."""
    f_name = data_loc + graph + ".edges"
    g = load_graph_from_csv(f_name, directed,
                            csv_options={'quotechar': '"',
                                         'delimiter': ' '})
    m, z = motif_significance(g, motif_size, n_shuffles,
                              shuffle_model=s_model)
    motif_annotation = str(motif_size) + 'm' if directed else str(motif_size) + 'um'
    output_name = "{}{}_{}.{}".format(data_loc, graph,
                                      motif_annotation, "motifslog")
    return write_motifs_results(output_name, m, z, n_shuffles, s_model)

def get_top_k(labels):
    """Return the number of classes for each row in the `labels`
    binary matrix. If `labels` is Linked List Matrix format, the number
    of labels is the length of each list, otherwise it is the number
    of non-zeros."""
    if isinstance(labels, csr_matrix):
        return [np.count_nonzero(i.toarray()) for i in labels]
    else:
        return [np.count_nonzero(i) for i in labels]


def run_embedding_classify_f1(dataset_name, emb_file, clf=LogisticRegression(),
                              splits_ratio=[0.5], num_run=2, write_to_file=None):
    """Run node classification for the learned embedding."""
    _, _, labels = load_data(dataset_name)
    emb = load_embeddings(emb_file)
    results_str = []
    averages = ["micro", "macro", "samples", "weighted"]
    for run in range(num_run):
        results_str.append("\nRun number {}:\n".format(run+1))
        for sr in splits_ratio:
            X_train, X_test, y_train, y_test = train_test_split(
                emb, labels, test_size=sr, random_state=run)
            top_k_list = get_top_k(y_test)
            mclf = TopKRanker(clf)
            mclf.fit(X_train, y_train)
            test_results = mclf.predict(X_test, top_k_list,
                                        num_classes=labels.shape[1])
            str_output = "Train ratio: {}\n".format(1.0 - sr)
            for avg in averages:
                str_output += avg + ': ' + str(f1_score(test_results, y_test,
                                                        average=avg)) + '\n'
            str_output += "Accuracy: " + str(accuracy_score(test_results, y_test)) + '\n'
            results_str.append(str_output)
    info = "Embedding dim: {}, graph: {}".format(emb.shape[1], dataset_name)
    if write_to_file:
        with open(write_to_file, 'w') as f:
            f.write(info)
            f.writelines(results_str)
    print(info)
    print(''.join(results_str))
    return write_to_file


class TopKRanker(OneVsRestClassifier):
    """Python 3 and sklearn 0.18.1 compatible version
    of the original implementation.
    https://github.com/gear/deepwalk/blob/master/example_graphs/scoring.py"""
    def predict(self, features, top_k_list, num_classes=39):
        """Predicts top k labels for each sample
        in the `features` list. `top_k_list` stores
        number of labels given in the dataset. This
        function returns a binary matrix containing
        the predictions."""
        assert features.shape[0] == len(top_k_list)
        probs = np.asarray(super().predict_proba(features))
        all_labels = np.zeros(shape=(features.shape[0], num_classes))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for l in labels:
                all_labels[i][l] = 1.0
        return all_labels


def significant_graph(motifslog, z_thres=100, m_size=3):
    """Plot and return the motifs significant graph from a logfile."""
    motif_zscore = dict()
    zscore = 0.0
    for line in open(motifslog):
        if re.match(r"Motif", line):
            zscore = float(line.split()[-1])
            if zscore > z_thres:
                pass


def tsne_visualization(emb_file, graph_labels, pdf_name, color_map=None, n_components=2):
    """Visualize the graph embedding with t-SNE"""
    emb = load_embeddings(emb_file)
    t0 = time()
    model = TSNE(n_components=n_components, init='pca', random_state=0)
    trans_data = model.fit_transform(embeddings_vectors).T
    t1 = time()
    print("t-SNE: %.2g sec" % (t1-t0))
    fig = plt.figure(figsize=(6.75,3))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(trans_data[0], trans_data[1], c=color_map)

