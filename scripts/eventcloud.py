import torch
import torch.nn as nn
import numpy as np

"""
To implement the following algorithm from Ren et al. CVPR 2024:

7: Hierarchy structure
8: for stage in range(Snum) do
9: Grouping and Sampling(P N)
10: Get P GS ∈ [B, Nstage, K, 2 * Dstage-1]
11: Local Extractor(P GS)
12: Get Flocal ∈ [B, Nstage, K, Dstage]
13: Attentive Aggregate(Flocal)
14: Get Faggre ∈ [B, Nstage, Dstage]
15: Global Extractor(Faggre)
16: Get P N = Fglobal ∈ [B, Nstage, Dstage]
17: end for
"""

class HierarchyStructure(nn.module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp,  group_all=False):
        super(HierarchyStructure, self).__init__()
        self.npoint = npoint
        self.radius