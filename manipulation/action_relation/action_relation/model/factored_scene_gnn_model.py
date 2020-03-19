import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import dgl

from itertools import permutations
import numpy as np


def prop_edge_data(edges):
    """Propagate data from edge to destination mailbox."""
    return {'m': edges.data['h'] }


def prop_node_data(edges):
    """Propagate data from edge source to destination mailbox."""
    return {'m': edges.src['h']}

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}


class GCNEdge(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCNEdge, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, edge_data=None):
        if edge_data is not None:
            g.edata['h'] = edge_data
        else:
            assert g.edata.get('h') is not None
        g.update_all(prop_edge_data, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, eh):
        g.ndata['h'] = eh
        g.update_all(prop_node_data, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(SimpleClassifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, e_features):
        h = e_features
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        output = self.classify(hg)
        output = self.sigmoid(output)
        return output
    

def get_inter_object_edges_based_on_z(obj_pos_tensor, num_objs):
    all_edges = list(permutations(range(num_objs), 2))
    valid_edges = []
    for e in all_edges:
        z1 = obj_pos_tensor[e[0], 2]
        z2 = obj_pos_tensor[e[1], 2]
        if abs(z1 - z2) < 0.035:
            valid_edges.append(all_edges)
    return valid_edges


class GNNNodeAndSceneClassifier(nn.Module):
    def __init__(self, inp_dim, args, use_backbone=True):
        super(GNNNodeAndSceneClassifier, self).__init__()
        self.inp_dim = inp_dim
        self.args = args
        self.use_backbone = use_backbone

        if use_backbone:
            self.backbone = nn.Sequential(
                nn.Linear(inp_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )

            gcn_inp_dim = 128
        else:
            print("==== Not using backbone ====")
            gcn_inp_dim = inp_dim

        self.add_obj_centers = False
        obj_center_size = 5 if self.add_obj_centers else 0

        if gcn_inp_dim >= 500:
            self.layers = nn.ModuleList([
                GCNEdge(gcn_inp_dim, 256 , F.relu),
                GCN(256 + obj_center_size, 64, F.relu),
                GCN(64, 16, F.relu)
                ])
        elif gcn_inp_dim >= 250:
            self.layers = nn.ModuleList([
                GCNEdge(gcn_inp_dim, 128, F.relu),
                GCN(128 + obj_center_size, 64, F.relu),
                GCN(64, 16, F.relu)
            ])
        elif gcn_inp_dim >= 120:
            node_inp_emb = 64 + 3 if self.add_obj_centers else 64
            self.layers = nn.ModuleList([
                GCNEdge(gcn_inp_dim, 64, F.relu),
                GCN(64 + obj_center_size, 32, F.relu),
                GCN(32, 16, F.relu)
            ])
        else:
            raise ValueError(f"Invalid input dim {in_dim}")

        self.node_classif = GCN(16, 2, None)
        self.graph_classif = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward_predict_precond(self, inp, bb_input=None):
        raise ValueError("Not implemented.")
    
    def forward_all_emb_with_backbone(self, emb):
        if self.use_backbone:
            return self.backbone(emb)
        else:
            return emb

    def forward_scene_emb_predict_precond(self, scene_emb_list, 
                                          scene_obj_pair_far_apart_list,
                                          bb_input=None,
                                          scene_obj_label_list=None,
                                          obj_center_tensor_list=None):
        if obj_center_tensor_list is not None:
            assert len(scene_emb_list) == len(obj_center_tensor_list)
        device = scene_emb_list[0][0].device
        graph_list = self.create_graph_list_for_scene_emb_list(
            scene_emb_list, 
            scene_obj_pair_far_apart_list
        )

        node_label_output_list = []
        graph_label_output_list = []
        for scene_i, all_pair_scene_emb in enumerate(scene_emb_list):
            x = None
            g = graph_list[scene_i]
            for l_idx, l in enumerate(self.layers):
                x = l(g, x)
                if l_idx == 0 and obj_center_tensor_list is not None and self.add_obj_centers:
                    obj_center_tensor = obj_center_tensor_list[scene_i].to(device)
                    x = torch.cat([x, obj_center_tensor], dim=1)

            all_node_x = torch.mean(x, dim=0, keepdim=True)
            graph_label = torch.sigmoid(self.graph_classif(all_node_x))

            node_label = self.node_classif(g, x)
            
            node_label_output_list.append(node_label)
            graph_label_output_list.append(graph_label)

        precond_output = torch.stack(graph_label_output_list).squeeze(2)
        return precond_output, node_label_output_list 

    def create_graph_list_for_scene_emb_list(self, 
                                             scene_emb_list, 
                                             scene_obj_pair_far_apart_list):
        graph_list = []
        for scene_i, all_pair_scene_emb in enumerate(scene_emb_list):
            num_objs = int((1 + np.sqrt(1 + 4 * len(all_pair_scene_emb))) / 2)
            all_edges = list(permutations(range(num_objs), 2))

            # Remove edges from graph which are far apart
            objs_far_apart = torch.stack(scene_obj_pair_far_apart_list[scene_i])
            objs_far_apart = objs_far_apart.cpu().numpy()
            assert len(all_edges) == len(objs_far_apart)
            all_edges = [e for i, e in enumerate(all_edges) if objs_far_apart[i] == 0]
            assert len(all_edges) > 0, "No object close to another object."

            src_edges = [e[0] for e in all_edges]
            dest_edges = [e[1] for e in all_edges] 
            num_nodes = num_objs
            g = dgl.DGLGraph()
            g.add_nodes(num_nodes)
            assert len(src_edges) == len(all_pair_scene_emb)
            g.add_edges(src_edges, dest_edges, {'h': torch.stack(all_pair_scene_emb)})

            graph_list.append(g)

        return graph_list


class GNNVolumeBasedNodeAndSceneClassifier(nn.Module):
    def __init__(self, inp_dim, args):
        super(GNNVolumeBasedNodeAndSceneClassifier, self).__init__()
        self.inp_dim = inp_dim
        self.args = args

        gcn_inp_dim = 5
        node_info_size = 11
        self.layers = nn.ModuleList([
            GCNEdge(gcn_inp_dim, 64, F.relu),
            GCN(64 + node_info_size, 64, F.relu),
            GCN(64, 32, F.relu),
            GCN(32, 16, F.relu)
        ])

        self.node_classif = GCN(16, 2, None)
        self.graph_classif = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward_predict_precond(self, inp, bb_input=None):
        raise ValueError("Not implemented.")
    
    def forward_all_emb_with_backbone(self, emb):
        return emb

    def forward_scene_emb_predict_precond(self, scene_emb_list, 
                                          scene_obj_pair_far_apart_list,
                                          bb_input=None,
                                          scene_obj_label_list=None,
                                          obj_info_tensor_list=None):
        assert obj_info_tensor_list is not None and \
               len(scene_emb_list) == len(obj_info_tensor_list)
        device = scene_emb_list[0][0].device
        graph_list = self.create_graph_list_for_scene_emb_list(
            scene_emb_list, 
            scene_obj_pair_far_apart_list,
            obj_info_tensor_list
        )

        node_label_output_list = []
        graph_label_output_list = []
        for scene_i, all_pair_scene_emb in enumerate(scene_emb_list):
            x = None
            g = graph_list[scene_i]
            for l_idx, l in enumerate(self.layers):
                x = l(g, x)
                if l_idx == 0:
                    obj_info_tensor = obj_info_tensor_list[scene_i].to(device)
                    x = torch.cat([x, obj_info_tensor], dim=1)

            all_node_x = torch.mean(x, dim=0, keepdim=True)
            graph_label = torch.sigmoid(self.graph_classif(all_node_x))

            node_label = self.node_classif(g, x)
            
            node_label_output_list.append(node_label)
            graph_label_output_list.append(graph_label)

        precond_output = torch.stack(graph_label_output_list).squeeze(2)
        return precond_output, node_label_output_list 

    def create_graph_list_for_scene_emb_list(self, 
                                             scene_emb_list, 
                                             scene_obj_pair_far_apart_list,
                                             scene_obj_info_list):
        graph_list = []
        device = scene_emb_list[0][0].device
        for scene_i, all_pair_scene_emb in enumerate(scene_emb_list):
            num_objs = int((1 + np.sqrt(1 + 4 * len(all_pair_scene_emb))) / 2)
            all_edges = list(permutations(range(num_objs), 2))
            obj_info_list = scene_obj_info_list[scene_i]

            valid_edges = get_inter_object_edges_based_on_z(obj_info_list, num_objs)

            obj_obj_rel_list = [obj_info_list[e[0]] - obj_info_list[e[1]] for e in all_edges]
            edge_data = torch.stack(obj_obj_rel_list).to(device)

            src_edges = [e[0] for e in all_edges]
            dest_edges = [e[1] for e in all_edges] 
            num_nodes = num_objs
            g = dgl.DGLGraph()
            g.add_nodes(num_nodes)
            assert len(src_edges) == len(all_pair_scene_emb)
            g.add_edges(src_edges, dest_edges, {'h': edge_data[:, :5]})

            import networkx as nx
            # Since the actual graph is undirected, we convert it for visualization
            # purpose.
            nx_G = g.to_networkx().to_undirected()
            # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
            pos = nx.kamada_kawai_layout(nx_G)
            nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
            import pdb; pdb.set_trace()

            graph_list.append(g)

        return graph_list
