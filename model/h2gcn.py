import torch
import torch.nn.functional as F
import torch_geometric.utils as gutils
from utils import remove_edges, gcn_norm, edge_index_to_sparse_tensor_adj


class H2GCNNet(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, layer_num=2, device='cpu'):
        super(H2GCNNet, self).__init__()

        self.linear1 = torch.nn.Linear(num_features, num_hidden)

        self.linear2 = torch.nn.Linear(num_hidden + 2 * num_hidden + 4 * num_hidden, num_classes)

        self.dropout = dropout
        self.layer_num = layer_num
        self.data = data
        self.device = device


        temp_loop_edge_index, _ = gutils.add_self_loops(self.data.edge_index)
        sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index, num_nodes=self.data.x.shape[0])


        k_hop_adjs = []
        k_hop_edge_index = []
        k_hop_adjs.append(sparse_adj_tensor)

        for i in range(self.layer_num-1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index



        for i in range(self.layer_num-1):
            self.k_hop_edge_index[i], _ = gutils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.data.edge_index).to(self.device)
            else:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.k_hop_edge_index[i-1]).to(self.device)

        self.norm_adjs = []
        self.norm_adjs.append(gcn_norm(self.data.edge_index, self.data.y.shape[0]))
        self.norm_adjs.append(gcn_norm(self.k_hop_edge_index[0], self.data.y.shape[0]))


    def forward(self, h, edge_index):
        h = self.linear1(self.data.x)
        h = F.relu(h)
        final_h = h

        # first layer
        first_hop_h = torch.sparse.mm(self.norm_adjs[0], h)
        second_hop_h = torch.sparse.mm(self.norm_adjs[1], h)
        R1 = torch.cat([first_hop_h, second_hop_h], dim=1)

        # second layer
        first_hop_h2 = torch.sparse.mm(self.norm_adjs[0], R1)
        second_hop_h2 = torch.sparse.mm(self.norm_adjs[1], R1)
        R2 = torch.cat([first_hop_h2, second_hop_h2], dim=1)

        final_h = torch.cat([final_h, R1], dim=1)
        final_h = torch.cat([final_h, R2], dim=1)
        final_h = F.dropout(final_h, p=self.dropout, training=self.training)
        final_h = self.linear2(final_h)

        return F.log_softmax(final_h, 1)

