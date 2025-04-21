import numpy as np
from typing import Optional

class KMeans:
    """
    K-Means clustering algorithm implementation.
    
    This implementation includes k-means++ initialization and supports
    multiple distance metrics for cluster assignment. It follows the
    scikit-learn API style for consistency.
    
    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters (K) to form and number of centroids to generate.
    
    max_iter : int, default=300
        Maximum number of iterations for a single run.
    
    tol : float, default=1e-4
        Tolerance for declaring convergence.
        If the change in cluster centroids is less than this value,
        the algorithm is considered to have converged.
    
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output in terms
        of inertia.
    
    random_state : Optional[int], default=None
        Seed for random number generation.
    
    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    
    n_iter_ : int
        Number of iterations run.
    """
    
    def __init__(self, n_clusters: Optional[int] = None, max_iter: int = 300,
                 tol: float = 1e-4, n_init: int = 10,
                 random_state: Optional[int] = None,
                 auto_k: bool = True, k_range: tuple = (2, 10)):
        self.n_clusters = n_clusters if n_clusters is not None else 8
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.auto_k = auto_k
        self.k_range = k_range
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.inertia_curve_ = None
    
    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """Initialize cluster centers using k-means++ algorithm.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances.
        
        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Initial cluster centers.
        """
        n_samples, n_features = X.shape
        centers = np.zeros((self.n_clusters, n_features))
        
        # Choose first center randomly
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        first_center_idx = np.random.randint(n_samples)
        centers[0] = X[first_center_idx]
        
        # Choose remaining centers
        for i in range(1, self.n_clusters):
            # Compute distances to existing centers
            distances = np.min([np.sum((X - center) ** 2, axis=1)
                              for center in centers[:i]], axis=0)
            
            # Choose next center with probability proportional to distance squared
            probabilities = distances / distances.sum()
            next_center_idx = np.random.choice(n_samples, p=probabilities)
            centers[i] = X[next_center_idx]
        
        return centers
    
    def _compute_labels_and_inertia(self, X: np.ndarray,
                                   centers: np.ndarray) -> tuple:
        """Compute cluster labels and inertia for given data and centers.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances.
        centers : array-like of shape (n_clusters, n_features)
            Cluster centers.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point.
        inertia : float
            Sum of squared distances to closest cluster center.
        """
        distances = np.array([np.sum((X - center) ** 2, axis=1)
                            for center in centers])
        labels = np.argmin(distances, axis=0)
        inertia = np.sum(np.min(distances, axis=0))
        
        return labels, inertia
    
    def _update_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update cluster centers based on current labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances.
        labels : array-like of shape (n_samples,)
            Current cluster labels.
        
        Returns
        -------
        new_centers : ndarray of shape (n_clusters, n_features)
            Updated cluster centers.
        """
        new_centers = np.array([X[labels == k].mean(axis=0)
                              if np.sum(labels == k) > 0 else self.cluster_centers_[k]
                              for k in range(self.n_clusters)])
        return new_centers
    
    def _calculate_inertia_curve(self, X: np.ndarray) -> tuple:
        """Calculate inertia for different values of k to find optimal number of clusters.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances.
        
        Returns
        -------
        optimal_k : int
            Optimal number of clusters determined by elbow method.
        inertias : list
            List of inertia values for different k.
        """
        inertias = []
        k_values = range(self.k_range[0], self.k_range[1] + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, max_iter=self.max_iter,
                          tol=self.tol, n_init=self.n_init,
                          random_state=self.random_state,
                          auto_k=False)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Calculate curvature to find elbow point
        x = np.array(list(k_values))
        y = np.array(inertias)
        curvature = np.gradient(np.gradient(y))
        optimal_k = x[np.argmax(np.abs(curvature))]        
        return optimal_k, inertias
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """Compute k-means clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.array(X)
        
        if self.auto_k and self.n_clusters is None:
            optimal_k, inertias = self._calculate_inertia_curve(X)
            self.n_clusters = optimal_k
            self.inertia_curve_ = inertias
        
        best_inertia = np.inf
        
        for init in range(self.n_init):
            # Initialize centers
            centers = self._kmeans_plus_plus_init(X)
            prev_centers = None
            n_iter = 0
            
            # Run k-means iterations
            while n_iter < self.max_iter:
                # Assign points to clusters
                labels, inertia = self._compute_labels_and_inertia(X, centers)
                
                # Update centers
                centers = self._update_centers(X, labels)
                
                # Check for convergence
                if prev_centers is not None:
                    center_shift = np.sum((centers - prev_centers) ** 2)
                    if center_shift < self.tol:
                        break
                
                prev_centers = centers.copy()
                n_iter += 1
            
            # Update best result
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.inertia_ = inertia
                self.n_iter_ = n_iter
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster for each sample in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        X = np.array(X)
        labels, _ = self._compute_labels_and_inertia(X, self.cluster_centers_)
        return labels
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_
    
    def plot(self, X: np.ndarray, plot_elbow: bool = False) -> None:
        """Plot the clustering results and optionally the elbow curve.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data points to plot.
        plot_elbow : bool, default=False
            Whether to plot the elbow curve if available.
        """
        from mlscratch.utils.visualization import Plotter
        if self.labels_ is None:
            raise ValueError("Model must be fitted before plotting.")
        
        if plot_elbow and self.inertia_curve_ is not None:
            k_values = range(self.k_range[0], self.k_range[1] + 1)
            Plotter.plot_elbow_curve(k_values, self.inertia_curve_)
        else:
            Plotter.plot_clusters(X, self.labels_, self.cluster_centers_)