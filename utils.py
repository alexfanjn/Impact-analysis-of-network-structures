import torch
import numpy as np
import scipy.sparse as sp
import random
import networkx as nx
import itertools
import torch_geometric

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix, degree
from copy import deepcopy
from torch_geometric.datasets import Planetoid

# calculate the accuracy metric
def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item()*1.0/len(labels)

# calculate the f1-score metric
def f1_scores(logits, labels):
    logits = torch.exp(logits).detach()
    pred = torch.argmax(logits, dim=1)
    final_f1_score = f1_score(y_true=labels, y_pred=pred)
    return final_f1_score

# train, val, test data splits
def train_val_test_split_tabular(N, train, test, val, stratify, random_state):

    idx = torch.arange(N)
    idx_train, idx_test = train_test_split(idx,
                                           random_state=random_state,
                                           train_size=train + val,
                                           test_size=test,
                                           stratify=stratify)
    if val:
        if stratify is not None:
            stratify = stratify[idx_train]
        idx_train, idx_val = train_test_split(idx_train,
                                              random_state=random_state,
                                              train_size=train / (train + val),
                                              stratify=stratify)
    else:
        idx_val = None

    return idx_train, idx_val, idx_test



# Remove edges in pyg data
# This function is adoped from the following link:
# https://github.com/EdisonLeeeee/GreatX/blob/master/greatx/utils/modification.py
def remove_edges(edge_index, edges_to_remove):
    edges_to_remove = torch.cat(
            [edges_to_remove, edges_to_remove.flip(0)], dim=1)
    edges_to_remove = edges_to_remove.to(edge_index)

    # it's not intuitive to remove edges from a graph represented as `edge_index`
    edge_weight_remove = torch.zeros(edges_to_remove.size(1)) - 1e5
    edge_weight = torch.cat(
        [torch.ones(edge_index.size(1)), edge_weight_remove], dim=0)
    edge_index = torch.cat([edge_index, edges_to_remove], dim=1).cpu().numpy()
    adj_matrix = sp.csr_matrix(
        (edge_weight.cpu().numpy(), (edge_index[0], edge_index[1])))
    adj_matrix.data[adj_matrix.data < 0] = 0.
    adj_matrix.eliminate_zeros()
    edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
    return edge_index


# from edge_index of pyg to sparse tensor adj data
def edge_index_to_sparse_tensor_adj(edge_index, num_nodes):
    sparse_adj_adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    values = sparse_adj_adj.data
    indices = np.vstack((sparse_adj_adj.row, sparse_adj_adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sparse_adj_adj.shape
    sparse_adj_adj_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return sparse_adj_adj_tensor


# normilize adj via (D^-0.5  A D^-0.5)
def gcn_norm(edge_index, num_nodes):
    a1 = edge_index_to_sparse_tensor_adj(edge_index, num_nodes)

    d1_adj = torch.diag(degree(edge_index[0], num_nodes=num_nodes)).to_sparse()
    d1_adj = torch.pow(d1_adj, -0.5)
    return torch.sparse.mm(torch.sparse.mm(d1_adj, a1), d1_adj)


# generate specific type of graph under homophily requirement in pyg format
def generate_pyg_graph(num_nodes, num_classes, graph_type, h, other_parameters):
    # generate labels
    num_nodes_per_class_list = []
    for i in range(num_classes):
        num_nodes_per_class = int(num_nodes / num_classes)
        num_nodes_per_class_list.append(num_nodes_per_class)

    if np.sum(num_nodes_per_class_list) != num_nodes:
        num_nodes_per_class_list[-1] = num_nodes - np.sum(num_nodes_per_class_list[:-1])

    labels = []
    for i in range(len(num_nodes_per_class_list)):
        for j in range(num_nodes_per_class_list[i]):
            labels.append(i)
    random.shuffle(labels)
    labels = torch.tensor(labels)


    # generate specific type of graph
    # the following generations are built from package networkx
    if graph_type == 'sf':
        m = other_parameters['m']
        # Add m initial nodes (m0 in barabasi-speak)
        G = nx.Graph()
        G.add_nodes_from(range(m))

        # Target nodes for new edges
        targets = list(range(m))

        # Start adding the other n-m nodes. The first node is m.
        source = m
        degree_list = []
        for i in range(m):
            degree_list.append(0)

        while source < num_nodes:
            # Add edges to m nodes from the source.
            G.add_edges_from(zip([source] * m, targets))

            if source == num_nodes - 1:
                break

            for i in range(len(targets)):
                degree_list[targets[i]] += 1

            degree_list.append(m)

            source += 1

            ### combine homophily h into the selection weights
            h_weight = np.zeros(source)
            h_weight[np.where(labels[:source] == labels[source])] = h
            h_weight[np.where(labels[:source] != labels[source])] = 1 - h
            final_weight = h_weight * degree_list

            final_weight_sum = np.sum(final_weight)

            targets = []
            while len(targets) < m:

                p = random.uniform(0, final_weight_sum)
                cnt = 0
                i = -1
                while cnt < p:
                    i += 1
                    cnt += final_weight[i]

                candidate_node = i
                if candidate_node not in targets:
                    targets.append(candidate_node)

    elif graph_type == 'sw':
        k = other_parameters['k']
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        nlist = list(G.nodes())

        fromv = nlist
        # connect the k/2 neighbors
        for j in range(1, k // 2 + 1):
            tov = fromv[j:] + fromv[0:j]  # the first j are now last
            for i in range(len(fromv)):
                G.add_edge(fromv[i], tov[i])
        # for each edge u-v, with probability p, randomly select existing
        # node w and add new edge u-w
        e = list(G.edges())
        for (u, v) in e:
            if random.random() < other_parameters['p']:
                same_label_list = list(np.array(torch.where(labels == labels[u])[0]))
                diff_label_list = list(np.array(torch.where(labels != labels[u])[0]))
                final_p = random.random()
                ### further selection of edge based on homophily
                if final_p > h:
                    w = random.choice(diff_label_list)
                else:
                    w = random.choice(same_label_list)

                # no self-loops and reject if edge u-w exists
                # is that the correct NWS model?
                while w == u or G.has_edge(u, w):
                    ### further selection of edge based on homophily
                    if final_p > h:
                        w = random.choice(diff_label_list)
                    else:
                        w = random.choice(same_label_list)
                    if G.degree(u) >= num_nodes - 1:
                        break  # skip this rewiring
                else:
                    G.add_edge(u, w)

    elif graph_type == 'er':
        edges = itertools.combinations(range(num_nodes), 2)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for e in edges:
            if random.random() < other_parameters['p']:
                final_p = random.random()
                ### further selection of edge based on homophily
                if labels[e[0]] == labels[e[1]]:
                    if final_p < h:
                        G.add_edge(*e)
                else:
                    if final_p > h:
                        G.add_edge(*e)
    else:
        print('Error graph type!')
        os._exit()

    # from networkx format to pyg format
    data = torch_geometric.utils.from_networkx(G)
    data.y = labels

    # print(torch_geometric.utils.homophily(data1.edge_index, data1.y))

    # sample features form real-world dataset based on labels
    class0_ids = torch.where(labels == 0)[0]
    class1_ids = torch.where(labels == 1)[0]

    ref_dataset = 'cora'
    sample_data = Planetoid(root=ref_dataset, name=ref_dataset)[0]
    sample_pool_0 = np.where(sample_data.y == 0)[0]
    sample_pool_1 = np.where(sample_data.y == 1)[0]
    sample_0_id = np.arange(sample_pool_0.shape[0])
    sample_1_id = np.arange(sample_pool_1.shape[0])
    class_0_id = np.random.choice(sample_0_id, class0_ids.shape[0])
    class_1_id = np.random.choice(sample_1_id, class1_ids.shape[0])
    class0_features = sample_data.x[class_0_id, :]
    class1_features = sample_data.x[class_1_id, :]
    feature_dim = sample_data.x.shape[1]



    class0_features = torch.tensor(class0_features).float()
    class1_features = torch.tensor(class1_features).float()


    x = torch.zeros((num_nodes, feature_dim))

    x[class0_ids, :] = deepcopy(class0_features)
    x[class1_ids, :] = deepcopy(class1_features)

    data.x = x

    return data, G


# fix seed to reproduce results
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    return seed
