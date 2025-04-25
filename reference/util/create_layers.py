from collections import namedtuple
from typing import List, Tuple
import numpy as np
import torch

LayerInfo = namedtuple("LayerInfo", ["nodes", "memories"])  # num nodes, num mems per node

def create_layers(layers_info_base: List[LayerInfo], fields: int, field_dim: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    layers = []
    layer_counts = []
    
    # Make a copy of layers_info to avoid modifying the input
    layers_info = [l for l in layers_info_base]
    
    for i in range(len(layers_info)):
        if i == 0:  # outer layer
            nodes, mems = layers_info[i]
            nodes = fields  # match the fields
            layers_info[i] = LayerInfo(nodes=nodes, memories=mems)
            dim = field_dim
            counts = torch.ones((nodes, mems))
            mem_matrix = torch.zeros((nodes, mems, dim)) + 0.5
            layers.append(mem_matrix)
            layer_counts.append(counts)
        else:
            nodes, mems = layers_info[i]
            prev_nodes, prev_mems = layers_info[i - 1]
            assert prev_nodes % nodes == 0, f"At layer {i}, prev is {prev_nodes} but nodes is {nodes}. Cannot divide evenly."
            children_per_node = prev_nodes // nodes
            counts = torch.ones((nodes, mems))
            mem_matrix = torch.zeros((nodes, children_per_node, mems, prev_mems))
            layers.append(mem_matrix)
            layer_counts.append(counts)
    
    return layers, layer_counts
