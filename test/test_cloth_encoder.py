import numpy as np
from torch_geometric.nn.pool import knn_graph, radius_graph
from torch import tensor

def test_encoder():
    DIM = 10

    np_cloth = np.arange(DIM*DIM*2).reshape(DIM*DIM, 2)
    for i in range(DIM*DIM):
        np_cloth[i] = [i//DIM, i%DIM]

    xy = tensor(np_cloth)
    # g = knn_graph(xy, 8, batch=None, loop=False, flow='target_to_source', num_workers=1)    
    g = radius_graph(xy, 1, batch=None, loop=False, flow='target_to_source', num_workers=1)    

    pass