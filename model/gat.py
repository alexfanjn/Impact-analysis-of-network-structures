from torch.nn import Module
import torch.nn.functional as F

from torch_geometric.nn import GATConv

class GATNet(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8, dropout=0.5)
        self.conv2 = GATConv(8 * hidden_dim, output_dim, heads=1, concat=False,
                             dropout=0.5)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
