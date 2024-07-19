import torch
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.utils import get_ppr
from sklearn.metrics import euclidean_distances

from ..utils import feature_propagation
from .base_modules import ActiveLearning


class RANDOM(ActiveLearning):
    def __init__(self, data, available_idx, budget):
        super(RANDOM, self).__init__(data, available_idx, budget)

    def _preprocessing(self):
        pass

    def _score_calculation(self, cur_cnt, train_idx):
        random_idx = self._available_idx[torch.randint(0, len(self._available_idx)-1, (1, )).item()]
        self._scores[random_idx] = 1


class AGE(ActiveLearning):
    def __init__(self, data, available_idx, budget, num_hops=2, basef=0.99):
        self._basef = basef
        self._num_hops = num_hops
        self._num_classes = data.y.max().item()+1
        self._linear = torch.nn.Linear(data.x.shape[1], self._num_classes)
        self._linear.reset_parameters()

        super(AGE, self).__init__(data, available_idx, budget)

    def _preprocessing(self):
        # density
        kmeans = KMeans(n_clusters=self._num_classes, random_state=0).fit(self._data.x)
        distance = euclidean_distances(self._data.x, kmeans.cluster_centers_)
        density = np.min(distance, axis=1)
        self._density = torch.tensor((density- density.min()) / (density.max() - density.min()), dtype=torch.float32)

        # centrality
        centrality = get_ppr(self._data.edge_index)[1]
        self._centrality = torch.tensor((centrality - centrality.min()) / (centrality.max() - centrality.min()))

        # feature propagation
        self._prop_feat = feature_propagation(self._data.clone(), self._num_hops)

    def _score_calculation(self, cur_cnt, train_idx):
        gamma = np.random.beta(1, 1.005-self._basef**cur_cnt)
        alpha = beta = (1-gamma) / 2
        # uncertainty
        uncertainty = torch.zeros(len(self._available_idx))
        if len(train_idx) != 0:
            optimizer = torch.optim.Adam(self._linear.parameters(), lr=0.1)
            self._linear.reset_parameters()
            self._linear.train()
            for _ in range(50):
                optimizer.zero_grad()
                out = self._linear(self._prop_feat)
                loss = torch.nn.CrossEntropyLoss()(out[train_idx], self._data.y[train_idx])
                loss.backward()
                optimizer.step()
            uncertainty = torch.nn.functional.softmax(out[self._available_idx], dim=1).max(dim=1).values.cpu().detach().numpy()
        self._uncertainty = torch.tensor((uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()))

        # unified score
        self._scores[self._available_idx] = alpha*self._uncertainty + \
            beta*self._density[self._available_idx] + gamma*self._centrality[self._available_idx]


class RIM(ActiveLearning):
    def __init__(self, data, available_idx, budget):
        super(RIM, self).__init__(data, available_idx, budget)

    def _preprocessing(self):
        pass

    def _score_calculation(self, cur_cnt, train_idx):
        pass
