import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

class BaseRandomForest:
    """Base class for random forest estimators.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    
    max_depth : int or None, default=None
        The maximum depth of each tree.
    
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    
    max_features : {"sqrt", "log2"} or int, default="sqrt"
        The number of features to consider when looking for the best split:
        - If int, then consider max_features features at each split.
        - If "sqrt", then max_features=sqrt(n_features).
        - If "log2", then max_features=log2(n_features).
    
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    
    n_jobs : int, default=1
        The number of jobs to run in parallel for both fit and predict.
        -1 means using all processors.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0,
                 max_features="sqrt", bootstrap=True, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.estimators_ = []
    
    def _get_n_features(self, n_features):
        """Get the number of features to consider for splitting."""
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        return n_features
    
    def _parallel_build_trees(self, X, y):
        """Build trees in parallel."""
        n_samples, n_features = X.shape
        n_features_split = self._get_n_features(n_features)
        
        def _build_tree(tree_idx):
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                sample_X = X[indices]
                sample_y = y[indices]
            else:
                sample_X = X
                sample_y = y
            
            tree = self._make_estimator()
            tree.max_features = n_features_split
            tree.fit(sample_X, sample_y)
            return tree
        
        n_jobs = min(self.n_jobs if self.n_jobs > 0 else os.cpu_count() or 1,
                    self.n_estimators)
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            self.estimators_ = list(executor.map(_build_tree, range(self.n_estimators)))

class RandomForestClassifier(BaseRandomForest):
    """A random forest classifier.
    
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split.
    
    max_depth : int or None, default=None
        The maximum depth of each tree.
    
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    
    max_features : {"sqrt", "log2"} or int, default="sqrt"
        The number of features to consider when looking for the best split.
    
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    
    n_jobs : int, default=1
        The number of jobs to run in parallel.
    """
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0,
                 max_features="sqrt", bootstrap=True, n_jobs=1):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs
        )
        self.criterion = criterion
    
    def _make_estimator(self):
        """Make and configure a copy of the base estimator."""
        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease
        )
    
    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.array(X)
        y = np.array(y)
        self._parallel_build_trees(X, y)
        return self
    
    def predict(self, X):
        """Predict class for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        X = np.array(X)
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.array([max(set(pred), key=list(pred).count)
                        for pred in predictions.T])

class RandomForestRegressor(BaseRandomForest):
    """A random forest regressor.
    
    A random forest is a meta estimator that fits a number of decision tree
    regressors on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    
    max_depth : int or None, default=None
        The maximum depth of each tree.
    
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    
    max_features : {"sqrt", "log2"} or int, default="sqrt"
        The number of features to consider when looking for the best split.
    
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    
    n_jobs : int, default=1
        The number of jobs to run in parallel.
    """
    def _make_estimator(self):
        """Make and configure a copy of the base estimator."""
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease
        )
    
    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.array(X)
        y = np.array(y)
        self._parallel_build_trees(X, y)
        return self
    
    def predict(self, X):
        """Predict regression target for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted values.
        """
        X = np.array(X)
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(predictions, axis=0)