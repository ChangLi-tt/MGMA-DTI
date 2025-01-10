import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from protein_encoder import Multi_order_gated
from multi_attention import Fea_extractor
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime


def binary_cross_entropy(pred_output, labels):  #BCE LOSS
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1) 
    loss = loss_fct(n, labels) 
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):  
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:

        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

def entropy_logits(linear_output):   
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class MGMADTI(nn.Module):
    def __init__(self, device='cuda', **config):
        super(MGMADTI, self).__init__()
        # Drug
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        # Protein
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"] 
        num_filters = config["PROTEIN"]["NUM_FILTERS"]     # [128, 128, 128]
        # MLP
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        # Padding
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]

        protein_num_head = config['PROTEIN']['NUM_HEAD']
        feature_num_head = config['CROSSINTENTION']['NUM_HEAD']
        feature_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        feature_layer = config['CROSSINTENTION']['LAYER']

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinModule(protein_emb_dim, num_filters, protein_num_head, protein_padding)

        self.feature_extraction = Fea_extractor(embed_dim=feature_emb_dim, num_head=feature_num_head, layer=feature_layer, device=device)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_extractor(v_p)
        f, v_d, v_p, att = self.feature_extraction(drug=v_d, protein=v_p)
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score  
        elif mode == "eval":
            return v_d, v_p, score, att 

# Drug feature extractor
class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)

        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):

        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        # batch_size
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

# Protein feature extractor
class ProteinModule(nn.Module):
    def __init__(self, embedding_dim, num_filters, padding=True):
        super(ProteinModule, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]  
        self.hor1 = Multi_order_gated(in_planes=in_ch[0], out_planes=in_ch[1]) 
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.hor2 = Multi_order_gated(in_planes=in_ch[1], out_planes=in_ch[2])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.hor3 = Multi_order_gated(in_planes=in_ch[2], out_planes=in_ch[3])
        self.bn3 = nn.BatchNorm1d(in_ch[3])
        

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.hor1(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn2(F.relu(self.hor2(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn3(F.relu(self.hor3(v.unsqueeze(-2))).squeeze(-2))
        v = v.view(v.size(0), v.size(2), -1)
        return v

# MLP
class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)

        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x))) 
        x = self.fc4(x)            
        return x
