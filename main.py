import torch_geometric
from utils import *
import torch.nn.functional as F

from model.gcn import GCNNet
from model.gat import GATNet
from model.fagcn import FAGCNNet
from model.h2gcn import H2GCNNet
import os



if __name__ == '__main__':

    # network setting
    num_nodes = 500
    avg_degree = 6
    num_classes = 2
    repeat_graph_generation_times = 1


    # tested graph types
    graph_type_list = ['er', 'sw', 'sf']

    # test gnns
    gnn_list = ['gcn', 'gat', 'fagcn', 'h2gcn']

    final_list = []


    # specific parameters of different graphs based on degree
    sf_setting = int(1 / 2 * avg_degree)
    sw_setting = int(2 / 3 * avg_degree)
    er_setting = avg_degree / (num_nodes - 1) * 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    for graph_type_id in range(len(graph_type_list)):

        for gnn_id in range(len(gnn_list)):
            # set seeds for generating multiple tested graphs
            graph_generation_seed_list = np.arange(repeat_graph_generation_times)

            for graph_generation_seed_id in range(len(graph_generation_seed_list)):
                # set seeds for generating multiple tested graphs
                set_seed(graph_generation_seed_list[graph_generation_seed_id])

                # select a specific graph type
                graph_type = graph_type_list[graph_type_id]

                # further setting of homophily on small-world networks due to its generation characteristic
                if graph_type == 'sw':
                    h = 0
                else:
                    h = 0.33


                gnn_type = gnn_list[gnn_id]

                if graph_type == 'sf':
                    other_parameters = {'m': sf_setting}
                elif graph_type == 'sw':
                    other_parameters = {'k': sw_setting, 'p': 0.5}
                elif graph_type == 'er':
                    other_parameters = {'p': er_setting}
                else:
                    print('Error graph type')
                    os._exit()

                # main function for generating specific graph
                data, G = generate_pyg_graph(num_nodes, num_classes, graph_type, h, other_parameters)

                print(data)
                print(f"Final h of generated graph: {torch_geometric.utils.homophily(data.edge_index, data.y)}")
                print(f"average degree: {torch.mean(torch_geometric.utils.degree(data.edge_index[0], num_nodes))}")


                # data splits
                train_ratio, test_ratio, val_ratio = 0.2, 0.6, 0.2
                lr = 0.001
                max_epoch = 500
                patience = 100
                min_loss = 100.0
                max_acc = 0


                idx_train, idx_val, idx_test = train_val_test_split_tabular(num_nodes, train_ratio, test_ratio,
                                                                            val_ratio, stratify=data.y,
                                                                            random_state=15)


                data = data.to(device)
                idx_train = idx_train.to(device)
                idx_val = idx_val.to(device)
                idx_test = idx_test.to(device)

                # gnn settings
                if gnn_type == 'gcn':
                    model = GCNNet(data.x.shape[1], 16, num_classes)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                elif gnn_type == 'gat':
                    model = GATNet(data.x.shape[1], 16, num_classes)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                elif gnn_type == 'fagcn':
                    dropout = 0.5
                    eps = 0.3
                    layer_num = 2
                    weight_decay = 5e-5

                    model = FAGCNNet(data, data.x.shape[1], 16, num_classes, dropout, eps, layer_num)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                elif gnn_type == 'h2gcn':
                    dropout = 0.5
                    layer_num = 2
                    weight_decay = 5e-5
                    data.edge_index = torch_geometric.utils.remove_self_loops(data.edge_index)[0]
                    model = H2GCNNet(data, data.x.shape[1], 16, num_classes, dropout, layer_num, device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                else:
                    pass


                model = model.to(device)

                # model training
                los = []
                for epoch in range(max_epoch):

                    model.train()
                    logp = model(data.x, data.edge_index)

                    cla_loss = F.nll_loss(logp[idx_train], data.y[idx_train])
                    loss = cla_loss
                    train_acc = accuracy(logp[idx_train], data.y[idx_train])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    logp = model(data.x, data.edge_index)
                    test_acc = accuracy(logp[idx_test], data.y[idx_test])
                    loss_val = F.nll_loss(logp[idx_val], data.y[idx_val]).item()
                    val_acc = accuracy(logp[idx_val], data.y[idx_val])
                    los.append([epoch, loss_val, val_acc, test_acc])

                    if loss_val < min_loss and max_acc < val_acc:
                        min_loss = loss_val
                        max_acc = val_acc
                        counter = 0
                    else:
                        counter += 1

                    if counter >= patience:
                        print('early stop')
                        break

                    if epoch % 50 == 0:
                        print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(
                            epoch, loss_val, train_acc, val_acc, test_acc))

                # model evaluation
                model.eval()
                logp = model(data.x, data.edge_index)
                final_test_acc = accuracy(logp[idx_test], data.y[idx_test])

                final_test_auc = auc(logp[idx_test].detach().cpu(), data.y[idx_test].cpu())

                final_test_f1_score = f1_scores(logp[idx_test].detach().cpu(), data.y[idx_test].cpu())

                print(f'{gnn_type} test acc {final_test_acc}')
                print(f'{gnn_type} test f1_score {final_test_f1_score}\n')
                temp_list = [final_test_acc, final_test_f1_score]
                final_list.append(temp_list)

    np.save("results.npy", np.array(final_list))

    print(np.array(final_list))



