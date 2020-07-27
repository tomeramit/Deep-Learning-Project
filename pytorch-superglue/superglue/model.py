import pdb
import torch_geometric
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn import MessagePassing
from torch.nn.parameter import Parameter
import torch
import time
from itertools import product, permutations
from superglue.sinkhorn import sinkhorn_pytorch

def generate_edges_intra(len1, len2):
    edges1 = torch.tensor(list(permutations(range(len1), 2)), dtype=torch.long, requires_grad=False).t().contiguous()
    edges2 = torch.tensor(list(permutations(range(len2), 2)), dtype=torch.long, requires_grad=False).t().contiguous() + len1
    edges = torch.cat([edges1, edges2], dim=1).cuda()
    return edges


def generate_edges_cross(len1, len2):
    edges1 = torch.tensor(list(product(range(len1), range(len1, len1+len2)))).t().contiguous()
    edges2 = torch.tensor(list(product(range(len1, len1+len2), range(len1)))).t().contiguous()
    edges = torch.cat([edges1, edges2], dim=1).cuda()
    return edges

class AttConv(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, **kwargs):
        super(AttConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.fc0 = torch.nn.Linear(self.heads * out_channels, self.heads * out_channels).cuda()
        self.fc1 = torch.nn.Linear(self.in_channels, self.heads * out_channels).cuda()
        self.fc2 = torch.nn.Linear(self.in_channels, self.heads * out_channels).cuda()
        self.fc3 = torch.nn.Linear(self.in_channels, self.heads * out_channels).cuda()

    def forward(self, x, edge_index, size=None):

        from torch_geometric.data import Batch, Data
        batch_list = []
        for i in range(x.shape[0]):
            batch_list.append(Data(x=x[i, :, :], edge_index=edge_index))
        batch = Batch.from_data_list(batch_list)

        q = self.fc1.forward(batch.x)
        k = self.fc2.forward(batch.x)
        v = self.fc3.forward(batch.x)

        return self.propagate(batch.edge_index, size=None, x=batch.x, q=q, k=k, v=v, batch=batch.batch)

    def message(self, q, k, v, v_i, v_j, q_i, q_j, k_i, k_j, edge_index):
        # Compute attention coefficients.
        # print(f"got {v_i.shape} {v_j.shape} {q_i.shape} {q_j.shape} {k_i.shape} {k_j.shape}")
        # pdb.set_trace()
        alpha = torch.nn.functional.softmax(q_i * k_j / 11.313708498984761, dim=1)
        m = alpha * v_j
        m = self.fc0.forward(m)
        return m

    def update(self, aggr_out):
        return aggr_out


class superglue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 64, bias=True).cuda()
        #self.bn1 = torch.nn.BatchNorm1d(num_features=50)

        self.fc2 = torch.nn.Linear(64, 128, bias=True).cuda()
        #self.bn2 = torch.nn.BatchNorm1d(num_features=50)

        self.mp1 = AttConv(in_channels=128, out_channels=128, heads=1)
        #self.bn3 = torch.nn.BatchNorm1d(num_features=100)

        self.mp2 = AttConv(in_channels=128, out_channels=128, heads=1)
        #self.bn4 = torch.nn.BatchNorm1d(num_features=100)

        self.mp3 = AttConv(in_channels=128, out_channels=128, heads=1)
        #self.bn5 = torch.nn.BatchNorm1d(num_features=100)

        self.mp4 = AttConv(in_channels=128, out_channels=128, heads=1)
        #self.bn6 = torch.nn.BatchNorm1d(num_features=100)

        self.fc3 = torch.nn.Linear(128, 128, bias=True).cuda()
        self.dustbin_weight = Parameter(torch.Tensor([0.9]).float().cuda(), requires_grad=True)

        self.edges_intra = generate_edges_intra(50, 50)
        self.edges_cross = generate_edges_cross(50, 50)

        self.mlp1 = torch.nn.Linear(256, 128, bias=True).cuda()
        self.mlp2 = torch.nn.Linear(256, 128, bias=True).cuda()
        self.mlp3 = torch.nn.Linear(256, 128, bias=True).cuda()
        self.mlp4 = torch.nn.Linear(256, 128, bias=True).cuda()


    def pos_encoder(self, p):
        x = self.fc1(p)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        return x

    def forward(self, p1, d1, p2, d2, matches):

        #print(p1.shape, d1.shape, p2[0,12,1])

        batch_size = p1.shape[0]

        px1 = self.pos_encoder(p1) + d1
        px2 = self.pos_encoder(p2) + d2

        x = torch.cat([px1, px2], dim=1)

        len1 = p1.shape[1]
        len2 = p2.shape[1]

        x1 = x + self.mlp1(torch.cat([x, self.mp1.forward(x, self.edges_intra).reshape(batch_size, x.shape[1], -1)], dim=2))
        x2 = x1 + self.mlp2(torch.cat([x1, self.mp2.forward(x1, self.edges_cross).reshape(batch_size, x.shape[1], -1)], dim=2))
        x3 = x2 + self.mlp3(torch.cat([x2, self.mp3.forward(x2, self.edges_intra).reshape(batch_size, x.shape[1], -1)], dim=2))
        x4 = x3 + self.mlp4(torch.cat([x3, self.mp4.forward(x3, self.edges_cross).reshape(batch_size, x.shape[1], -1)], dim=2))

        x5 = torch.nn.functional.relu(self.fc3.forward(x4))

        #p11 = torch.nn.functional.relu(self.fc3.forward(p1))
        #p12 = torch.nn.functional.relu(self.fc3.forward(p1))
        #p11 = p11 / torch.norm(p11, dim=2, keepdim=True)
        x5 = x5 / torch.norm(x5, dim=2, keepdim=True)
        v1 = x5[:, :len1, :]
        v2 = x5[:, len1:, :]

        #pdb.set_trace()

        #v1 = p11[0, :, :]
        #v2 = p12[0, :, :]

        costs = torch.bmm(v1, v2.permute(0, 2, 1))

        dustbin_x = torch.ones((batch_size, 1, costs.shape[1])).cuda() * self.dustbin_weight
        dustbin_y = torch.ones((batch_size, costs.shape[1] + 1, 1)).cuda() * self.dustbin_weight
        costs_x = torch.cat([costs, dustbin_x], dim=1)
        costs_with_dustbin = torch.cat([costs_x, dustbin_y], dim=2)

        costs_with_dustbin2 = 1 + (-costs_with_dustbin)  # / costs_with_dustbin.sum()

        #print(costs_with_dustbin2)

        n1 = torch.ones((batch_size, len1 + 1), requires_grad=False).cuda()
        n2 = torch.ones((batch_size, len2 + 1), requires_grad=False).cuda()

        n1[:, -1] = len2
        n2[:, -1] = len1

        sol = sinkhorn_pytorch(n1, n2, costs_with_dustbin2, reg=0.01)

        #print(self.dustbin_weight)
        #print(f"??? {sol[:,:50, :50].mean()} | {costs.mean()}")
        #print(torch.where(sol[:,:50,:50] > 0.5)[0].shape)
        if self.training:
            loss = []
            acc = []
            for batch_idx in range(len(matches)):
                loss_for_batch_idx = []
                acc_for_batch_idx = []
                for match in matches[batch_idx].T:
                    #print(match[0], match[1])
                    loss_for_batch_idx.append(-torch.log(sol[batch_idx, match[0], match[1]] + 1e-5).reshape(-1))
                    #print(-torch.log(sol[batch_idx, match[0], match[1]] + 1e-5).reshape(-1))
                    #acc.append((torch.argmax(sol[match[0], :]) == match[1]).reshape(-1).float())
                    acc_for_batch_idx.append(((sol[batch_idx, match[0], match[1]]) >= 0.5).reshape(-1).float())
                loss.append(torch.cat(loss_for_batch_idx).mean().reshape(-1))
                acc.append(torch.cat(acc_for_batch_idx).mean().reshape(-1))

            loss = torch.cat(loss).mean()
            acc = torch.cat(acc).mean()

            return loss, acc
        else:
            return sol