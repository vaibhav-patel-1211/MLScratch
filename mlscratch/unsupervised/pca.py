import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implementation.
    
    PCA performs dimensionality reduction by projecting the data onto principal
    components that maximize variance. This implementation includes data
    standardization, covariance matrix computation, and eigenvalue decomposition.
    
    Parameters
    ----------
    n_components : int or float, default=None
        Number of components to keep:
        - If n_components is None, keep all components
        - If int, n_components must be <= n_features
        - If float between 0 and 1, select the number of components such that
          the amount of variance that needs to be explained is greater than
          the percentage specified
    
    standardize : bool, default=True
        Whether to standardize the input data to zero mean and unit variance.
    
    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space.
    
    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
    
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
    
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, used for centering the data.
    
    scale_ : ndarray of shape (n_features,)
        Per-feature scale used for normalization.
    """
    
    def __init__(self, n_components=None, standardize=True):
        self.n_components = n_components
        self.standardize = standardize
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.scale_ = None
    
    def _standardize_data(self, X):
        """Standardize the data to have zero mean and unit variance."""
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        if self.standardize:
            self.scale_ = np.std(X_centered, axis=0, ddof=1)
            # Handle constant features
            self.scale_[self.scale_ == 0] = 1
            X_scaled = X_centered / self.scale_
        else:
            self.scale_ = np.ones(X.shape[1])
            X_scaled = X_centered
            
        return X_scaled
    
    def fit(self, X):
        """Fit the PCA model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Standardize the data
        X_scaled = self._standardize_data(X)
        
        # Compute covariance matrix
        cov_matrix = np.dot(X_scaled.T, X_scaled) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determine number of components
        if self.n_components is None:
            n_components = n_features
        elif isinstance(self.n_components, float):
            # Select components based on explained variance ratio
            total_var = eigenvalues.sum()
            ratio_cumsum = np.cumsum(eigenvalues) / total_var
            n_components = np.sum(ratio_cumsum <= self.n_components) + 1
        else:
            n_components = min(self.n_components, n_features)
        
        # Store results
        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = eigenvalues[:n_components]
        self.explained_variance_ratio_ = eigenvalues[:n_components] / eigenvalues.sum()
        
        return self
    
    def transform(self, X):
        """Apply dimensionality reduction to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X = np.array(X)
        X_scaled = (X - self.mean_) / self.scale_
        return np.dot(X_scaled, self.components_.T)
    
    def fit_transform(self, X):
        """Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Transform data back to its original space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in transformed space.
        
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data in original space.
        """
        X_original = np.dot(X, self.components_)
        return X_original * self.scale_ + self.mean_