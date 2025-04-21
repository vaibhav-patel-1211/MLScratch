import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any

class Plotter:
    """Base class for visualization functionality.
    
    This class provides common plotting methods that can be used across different
    machine learning algorithms for visualizing their results and performance metrics.
    """
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], title: str = 'Training History',
                            xlabel: str = 'Iteration', ylabel: str = 'Value') -> None:
        """Plot training metrics over iterations.
        
        Parameters
        ----------
        history : Dict[str, List[float]]
            Dictionary containing metrics history (e.g., {'loss': [...], 'accuracy': [...]})
        title : str, default='Training History'
            Title of the plot
        xlabel : str, default='Iteration'
            Label for x-axis
        ylabel : str, default='Value'
            Label for y-axis
        """
        plt.figure(figsize=(10, 6))
        for metric_name, values in history.items():
            plt.plot(values, label=metric_name)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_scatter(X: np.ndarray, y: Optional[np.ndarray] = None,
                    title: str = 'Data Distribution', xlabel: str = 'X',
                    ylabel: str = 'Y') -> None:
        """Create scatter plot of data points.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data points to plot
        y : array-like of shape (n_samples,), optional
            Labels or target values for coloring the points
        title : str, default='Data Distribution'
            Title of the plot
        xlabel : str, default='X'
            Label for x-axis
        ylabel : str, default='Y'
            Label for y-axis
        """
        plt.figure(figsize=(10, 6))
        if X.shape[1] > 2:
            print("Warning: Only first two dimensions will be plotted")
            X = X[:, :2]
        
        if y is not None:
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
            plt.colorbar(label='Target')
        else:
            plt.scatter(X[:, 0], X[:, 1])
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(X: np.ndarray, y: np.ndarray,
                             predict_func: callable,
                             title: str = 'Decision Boundary') -> None:
        """Plot decision boundary for classification algorithms.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        predict_func : callable
            Function that makes predictions on input data
        title : str, default='Decision Boundary'
            Title of the plot
        """
        if X.shape[1] != 2:
            raise ValueError("Decision boundary plotting only supports 2D input")
        
        plt.figure(figsize=(10, 6))
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # Make predictions on mesh grid points
        Z = predict_func(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and points
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    
    @staticmethod
    def plot_clusters(X: np.ndarray, labels: np.ndarray,
                     centers: Optional[np.ndarray] = None,
                     title: str = 'Cluster Assignment') -> None:
        """Plot cluster assignments and centroids for clustering algorithms.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data points
        labels : array-like of shape (n_samples,)
            Cluster labels for each point
        centers : array-like of shape (n_clusters, n_features), optional
            Coordinates of cluster centers
        title : str, default='Cluster Assignment'
            Title of the plot
        """
        if X.shape[1] > 2:
            print("Warning: Only first two dimensions will be plotted")
            X = X[:, :2]
            if centers is not None:
                centers = centers[:, :2]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], c='red',
                       marker='x', s=200, linewidths=3,
                       label='Centroids')
            plt.legend()
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()