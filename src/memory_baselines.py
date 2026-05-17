import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class RawMemoryFull:
    def __init__(self, k=5):
        self.k = k
        self._X = None
        self._y_m1 = None
        self._y_0 = None
        self._y_p1 = None
        self._knn_m1 = None
        self._knn_0 = None
        self._knn_p1 = None

    @property
    def stored_samples_count(self):
        return len(self._X) if self._X is not None else 0

    @staticmethod
    def cost_bytes(n_features, n_samples):
        return n_features * n_samples * 4

    def fit(self, X, outcomes):
        self._X = np.array(X)
        if isinstance(outcomes[0], (list, np.ndarray)) and len(outcomes) == 3:
            self._y_m1 = np.array(outcomes[0]).ravel()
            self._y_0 = np.array(outcomes[1]).ravel()
            self._y_p1 = np.array(outcomes[2]).ravel()
        else:
            self._y_m1 = np.array([o[-1] for o in outcomes])
            self._y_0 = np.array([o[0] for o in outcomes])
            self._y_p1 = np.array([o[1] for o in outcomes])
        self._knn_m1 = KNeighborsRegressor(n_neighbors=min(self.k, len(X)), metric="euclidean")
        self._knn_0 = KNeighborsRegressor(n_neighbors=min(self.k, len(X)), metric="euclidean")
        self._knn_p1 = KNeighborsRegressor(n_neighbors=min(self.k, len(X)), metric="euclidean")
        self._knn_m1.fit(self._X, self._y_m1)
        self._knn_0.fit(self._X, self._y_0)
        self._knn_p1.fit(self._X, self._y_p1)

    def predict(self, X):
        X_arr = np.array(X)
        y_m1 = self._knn_m1.predict(X_arr)
        y_0 = self._knn_0.predict(X_arr)
        y_p1 = self._knn_p1.predict(X_arr)
        return np.stack([y_m1, y_0, y_p1], axis=-1)


class RawMemoryEqualCost:
    def __init__(self, param_budget=5000, k=5):
        self.param_budget = param_budget
        self.k = k
        self._X = None
        self._y_m1 = None
        self._y_0 = None
        self._y_p1 = None

    @property
    def stored_samples_count(self):
        return len(self._X) if self._X is not None else 0

    def cost_bytes(self):
        if self._X is None:
            return 0
        return self._X.shape[1] * len(self._X) * 4

    def fit(self, X, outcomes):
        n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
        max_samples = self.param_budget // (n_features * 4)
        n_samples = min(max_samples, len(X))
        idxs = np.random.default_rng(42).choice(len(X), n_samples, replace=False)
        self._X = np.array([X[i] for i in idxs])
        if isinstance(outcomes[0], (list, np.ndarray)) and len(outcomes) == 3:
            self._y_m1 = np.array(outcomes[0]).ravel()[idxs]
            self._y_0 = np.array(outcomes[1]).ravel()[idxs]
            self._y_p1 = np.array(outcomes[2]).ravel()[idxs]
        else:
            self._y_m1 = np.array([outcomes[i][-1] for i in idxs])
            self._y_0 = np.array([outcomes[i][0] for i in idxs])
            self._y_p1 = np.array([outcomes[i][1] for i in idxs])
        self._knn_m1 = KNeighborsRegressor(n_neighbors=min(self.k, n_samples))
        self._knn_0 = KNeighborsRegressor(n_neighbors=min(self.k, n_samples))
        self._knn_p1 = KNeighborsRegressor(n_neighbors=min(self.k, n_samples))
        self._knn_m1.fit(self._X, self._y_m1)
        self._knn_0.fit(self._X, self._y_0)
        self._knn_p1.fit(self._X, self._y_p1)

    def predict(self, X):
        X_arr = np.array(X)
        y_m1 = self._knn_m1.predict(X_arr)
        y_0 = self._knn_0.predict(X_arr)
        y_p1 = self._knn_p1.predict(X_arr)
        return np.stack([y_m1, y_0, y_p1], axis=-1)


class PrototypeMemory:
    def __init__(self, n_clusters=20, k=3):
        self.n_clusters = n_clusters
        self.k = k
        self._centroids = None
        self._y_m1 = None
        self._y_0 = None
        self._y_p1 = None
        self.labels_ = None

    @property
    def stored_samples_count(self):
        return self.n_clusters

    def cost_bytes(self):
        if self._centroids is None:
            return 0
        return self._centroids.shape[1] * self.n_clusters * 4

    def fit(self, X, outcomes):
        from sklearn.cluster import KMeans
        X_arr = np.array(X)
        nc = min(self.n_clusters, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        self.labels_ = km.fit_predict(X_arr)
        self._centroids = km.cluster_centers_
        self._y_m1 = np.zeros(nc)
        self._y_0 = np.zeros(nc)
        self._y_p1 = np.zeros(nc)
        if isinstance(outcomes[0], (list, np.ndarray)) and len(outcomes) == 3:
            om1 = np.array(outcomes[0]).ravel()
            o0 = np.array(outcomes[1]).ravel()
            op1 = np.array(outcomes[2]).ravel()
            for i in range(nc):
                mask = self.labels_ == i
                if mask.sum() > 0:
                    self._y_m1[i] = np.mean(om1[mask])
                    self._y_0[i] = np.mean(o0[mask])
                    self._y_p1[i] = np.mean(op1[mask])
        else:
            for i in range(nc):
                mask = self.labels_ == i
                if mask.sum() > 0:
                    self._y_m1[i] = np.mean([outcomes[j][-1] for j in np.where(mask)[0]])
                    self._y_0[i] = np.mean([outcomes[j][0] for j in np.where(mask)[0]])
                    self._y_p1[i] = np.mean([outcomes[j][1] for j in np.where(mask)[0]])

    def predict(self, X):
        from sklearn.metrics import pairwise_distances_argmin_min
        X_arr = np.array(X)
        idxs, _ = pairwise_distances_argmin_min(X_arr, self._centroids)
        y_m1 = self._y_m1[idxs]
        y_0 = self._y_0[idxs]
        y_p1 = self._y_p1[idxs]
        return np.stack([y_m1, y_0, y_p1], axis=-1)