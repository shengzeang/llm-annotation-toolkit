import torch
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.utils import get_ppr
from sklearn.metrics import euclidean_distances
from sklearn import cluster
import networkx as nx

from ..utils import feature_propagation, GraphPartition
from .base_modules import ActiveLearning


class RANDOM(ActiveLearning):
    def __init__(self, data, available_idx, budget):
        super(RANDOM, self).__init__(data, available_idx, budget)

    def _preprocessing(self):
        pass

    def _score_calculation(self, cur_cnt, train_idx):
        # the bridge between is the available_idx, rather implicit
        random_idx = self._available_idx[torch.randint(0, len(self._available_idx)-1, (1, )).item()]
        self._scores[random_idx] = 1


class DEGREE(ActiveLearning):
    def __init__(self, data, available_idx, budget):
        super(DEGREE, self).__init__(data, available_idx, budget)

    def _preprocessing(self):
        self.degree = torch.bincount(self._data.edge_index[1])

    def _score_calculation(self, cur_cnt, train_idx):
        self._scores[self._available_idx] = self.degree[self._available_idx]


class DENSITY(ActiveLearning):
    def __init__(self, data, available_idx, budget):
        super(DENSITY, self).__init__(data, available_idx, budget)

    def _preprocessing(self):
        num_classes = self._data.y.max().item() + 1
        # first conduct clustering
        model = cluster.KMeans(n_clusters=num_classes, init='k-means++', random_state=42)
        model.fit(self._data.x)
        # then calculate density
        centers = model.cluster_centers_
        label = model.predict(self._data.x)
        centers = centers[label]
        dist_map = torch.linalg.norm(self._data.x - centers, dim=1)
        dist_map = torch.tensor(dist_map, dtype=self._data.x.dtype, device=self._data.x.device)
        self.density = 1 / (1 + dist_map)

    def _score_calculation(self, cur_cnt, train_idx):
        self._scores[self._available_idx] = self.density[self._available_idx]


class PAGERANK(ActiveLearning):
    def __init__(self, data, available_idx, budget):
        super(PAGERANK, self).__init__(data, available_idx, budget)

    def _preprocessing(self):
        num_nodes = self._data.x.shape[0]
        self.pagerank = get_ppr(self._data.edge_index, num_nodes=num_nodes)[1]

    def _score_calculation(self, cur_cnt, train_idx):
        self._scores[self._available_idx] = self.pagerank[self._available_idx]


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


class FEATPROP(ActiveLearning):
    def __init__(self, data, available_idx, budget, num_hops=2):
        self._num_hops = num_hops
        self._num_classes = data.y.max().item()+1

        super(FEATPROP, self).__init__(data, available_idx, budget)

    def _preprocessing(self):
        # feature propagation
        self._prop_feat = feature_propagation(self.data.clone(), self._num_hops)
        # calculate pair-wise l2 distance
        self._l2_dist = torch.cdist(self._prop_feat, self._prop_feat, p=2)

    def _score_calculation(self, cur_cnt, train_idx):
        self._scores[self._available_idx] = self._l2_dist[self._available_idx]


class GPART(ActiveLearning):
    def __init__(self, data, available_idx, budget, num_hops=2):
        self._num_hops = num_hops
        self._num_classes = data.y.max().item()+1

        super(GPART, self).__init__(data, available_idx, budget)

    def split_cluster(budget, num_parts):
        part_size = [budget // num_parts for _ in range(num_parts)]
        for i in range(budget % num_parts):
            part_size[i] += 1
        return part_size

    def _preprocessing(self):
        # execute K graph partitions, K equals to the number of classes
        num_classes = self._data.y.max().item() + 1
        g = nx.Graph()
        edges = [(i.item(), j.item()) for i, j in zip(self._data.edge_index[0], self._data.edge_index[1])]
        g.add_edges_from(edges)
        graph = g.to_undirected()
        graph_part = GraphPartition(graph, self._data.x, num_classes)
        communities = graph_part.clauset_newman_moore(weight=None)
        sizes = ([len(com) for com in communities])
        threshold = 1 / 3
        if min(sizes) * len(sizes) / len(self._data.x) < threshold:
            partitions = graph_part.agglomerative_clustering(communities)
        else:
            sorted_communities = sorted(communities, key=lambda c: len(c), reverse=True)
            partitions = {}
            partitions[len(sizes)] = torch.zeros(self._data.x.shape[0], dtype=torch.int)
            for i, com in enumerate(sorted_communities):
                partitions[len(sizes)][com] = i
        self.partitions = partitions
        # pre-propagate the features in advance
        self.prop_x = feature_propagation(self._data.clone(), self._num_hops)

        

    def _score_calculation(self, cur_cnt, train_idx):
        # calculate cluster centroids within each pre-partitioned graph partitions
        num_nodes = self._data.x.shape[0]
        num_classes = self._data.y.max().item() + 1
        num_parts = int(np.ceil(self._budget / num_classes))

        partitions = np.array(partitions[num_parts].cpu())
        part_size = self.split_cluster(self._budget, num_parts)

        # exclude those nodes that have already been selected in previous rounds
        available_mask = torch.zeros(num_nodes, dtype=torch.bool)
        available_mask[self._available_idx] = True

        # Iterate over each partition
        indices = []
        for i in range(num_parts):
            part_id = np.where(partitions == i and available_mask == True)[0]
            masked_id = [i for i, x in enumerate(part_id) if x in indices]
            xi = self._data.x[part_id]

            n_clusters = part_size[i]
            if n_clusters <= 0:
                continue

            # Perform K-Means clustering:
            kmeans = cluster.KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
            kmeans.fit(xi.cpu().numpy())
            centers = kmeans.cluster_centers_

            # Obtain the centers
            for center in centers:
                center = torch.tensor(center, dtype=self._data.x.dtype, device=self._data.x.device)
                dist_map = torch.linalg.norm(xi - center, dim=1)
                dist_map[masked_id] = torch.tensor(np.infty, dtype=dist_map.dtype, device=dist_map.device)
                idx = int(torch.argmin(dist_map))
                masked_id.append(idx)
                indices.append(part_id[idx])
        
        self._scores[indices] = 1


class RIM(ActiveLearning):
    def __init__(self, data, available_idx, budget):
        super(RIM, self).__init__(data, available_idx, budget)

    def _preprocessing(self):
        pass

    def _score_calculation(self, cur_cnt, train_idx):
        pass
