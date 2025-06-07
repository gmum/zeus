import numpy as np
from scipy.special import logsumexp


class SimplifiedGMM:
    """
    Simplified Gaussian Mixture Model clustering algorithm.

    This implementation assumes diagonal fixed identity covariance matrices
    for all components, providing a simpler version of GMM while maintaining
    a scikit-learn-like API.

    Parameters
    ----------
    n_components : int, default=2
        Number of Gaussian components in the mixture model.

    max_iter : int, default=100
        Maximum number of iterations for the EM algorithm.

    tol : float, default=1e-4
        Convergence threshold for the log-likelihood.

    random_state : int or None, default=None
        Random seed for initialization.

    n_init : int, default=10
        Number of initializations to perform. The best results are kept.
    """

    def __init__(self, n_components=2, max_iter=100, tol=1e-4, random_state=None, n_init=10):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init

        # Model parameters to be learned
        self.means_ = None  # Component means
        self.weights_ = None  # Component weights/mixing coefficients

        # Additional attributes
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -np.inf

    def _initialize_params(self, X, rng):
        """Initialize the model parameters for a single run."""
        n_samples, n_features = X.shape

        # Randomly initialize means by picking random data points
        idx = rng.choice(n_samples, self.n_components, replace=False)
        means = X[idx].copy()

        # Initialize weights to be uniform
        weights = np.ones(self.n_components) / self.n_components

        return means, weights

    def fit(self, X):
        """
        Fit the GMM to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)

        # Run multiple initializations and keep the best one
        max_lower_bound = -np.inf
        best_params = None
        rng = np.random.RandomState(self.random_state)

        for init in range(self.n_init):
            # Initialize parameters
            means, weights = self._initialize_params(X, rng)

            # Run a single EM algorithm
            means, weights, lower_bound, converged, n_iter = self._fit_single(X, means, weights)

            # Update best parameters if needed
            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = (means, weights, lower_bound, converged, n_iter)

        # Set the best parameters
        self.means_, self.weights_, self.lower_bound_, self.converged_, self.n_iter_ = best_params

        return self

    def _fit_single(self, X, means, weights):
        """Run a single EM algorithm."""
        n_samples = X.shape[0]
        prev_lower_bound = -np.inf

        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities
            responsibilities = self._e_step(X, means, weights)

            # Check for collapsed clusters and reinitialize if necessary
            n_resp = responsibilities.sum(axis=0)
            if np.any(n_resp < 10):  # Consider clusters with few assigned points as collapsed
                # Reset any collapsed cluster to a random data point
                for k in np.where(n_resp < 10)[0]:
                    idx = np.random.randint(n_samples)
                    means[k] = X[idx].copy()
                    weights = np.ones(self.n_components) / self.n_components

                # Recompute responsibilities with new means
                responsibilities = self._e_step(X, means, weights)

            # M-step: Update parameters
            means, weights = self._m_step(X, responsibilities)

            # Compute lower bound
            # eps = 1e-8
            weighted_log_prob = self._estimate_weighted_log_prob(X, means, weights)
            lower_bound = np.sum(logsumexp(weighted_log_prob, axis=1))

            # Check for convergence
            change = lower_bound - prev_lower_bound
            if abs(change) < self.tol:
                converged = True
                break

            prev_lower_bound = lower_bound
            n_iter = iteration + 1
        else:
            converged = False
            n_iter = self.max_iter

        return means, weights, lower_bound, converged, n_iter

    def _e_step(self, X, means=None, weights=None):
        """E-step: compute responsibilities."""
        if means is None:
            means = self.means_
        if weights is None:
            weights = self.weights_

        weighted_log_prob = self._estimate_weighted_log_prob(X, means, weights)

        # Numerical stability: subtract max from each row
        log_prob_norm = logsumexp(weighted_log_prob, axis=1, keepdims=True)
        log_resp = weighted_log_prob - log_prob_norm
        return np.exp(log_resp)

    def _m_step(self, X, responsibilities):
        """M-step: update parameters."""
        n_samples = X.shape[0]

        # Update weights (mixing coefficients)
        sum_resp = responsibilities.sum(axis=0)
        weights = sum_resp / n_samples

        # Update means
        weighted_X = np.dot(responsibilities.T, X)
        means = weighted_X / sum_resp[:, np.newaxis]

        return means, weights

    def _estimate_weighted_log_prob(self, X, means=None, weights=None):
        """Compute the weighted log probabilities for each sample."""
        if means is None:
            means = self.means_
        if weights is None:
            weights = self.weights_

        n_samples, n_features = X.shape
        n_components = len(weights)
        log_prob = np.zeros((n_samples, n_components))

        # Use identity covariance matrix for all components
        for k in range(n_components):
            # Log probability for multivariate normal with identity covariance
            diff = X - means[k]
            log_prob[:, k] = -0.5 * np.sum(diff * diff, axis=1) - 0.5 * n_features * np.log(2 * np.pi)

        return log_prob + np.log(weights)

    def predict(self, X):
        """
        Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        if self.means_ is None:
            raise ValueError("Model not fitted yet.")
        X = np.asarray(X)
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """
        Predict posterior probability of each component given the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Posterior probabilities of each component for each sample.
        """
        if self.means_ is None:
            raise ValueError("Model not fitted yet.")
        X = np.asarray(X)
        return self._e_step(X)

    def sample(self, n_samples=1):
        """
        Generate random samples from the fitted Gaussian mixture distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.
        """
        if self.means_ is None:
            raise ValueError("Model not fitted yet.")

        n_features = self.means_.shape[1]
        rng = np.random.RandomState(self.random_state)

        # Choose which component to sample from
        component_indices = rng.choice(
            self.n_components, size=n_samples, p=self.weights_
        )

        # Sample from each selected component (identity covariance)
        X = np.zeros((n_samples, n_features))
        for i, component_idx in enumerate(component_indices):
            # Draw from normal with identity covariance
            X[i] = rng.multivariate_normal(
                mean=self.means_[component_idx],
                cov=np.eye(n_features)
            )

        return X

    def score_samples(self, X):
        """
        Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample under the model.
        """
        X = np.asarray(X)
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        return logsumexp(weighted_log_prob, axis=1)

    def bic(self, X):
        """
        Bayesian Information Criterion for the current model on the input X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.

        Returns
        -------
        bic : float
            The lower the better.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Free parameters: means and weights (minus 1 for sum constraint)
        n_params = self.n_components * n_features + (self.n_components - 1)

        log_likelihood = np.sum(self.score_samples(X))
        return -2 * log_likelihood + n_params * np.log(n_samples)