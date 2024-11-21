import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import cvxopt
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    weights: np.ndarray
    objective_value: float
    success: bool
    message: str

class PortfolioOptimizer:
    """Advanced portfolio optimization strategies."""
    
    def __init__(self, returns: np.ndarray, cov_matrix: np.ndarray):
        """
        Initialize the optimizer.
        
        Args:
            returns: Array of historical returns (n_assets, n_periods)
            cov_matrix: Covariance matrix of returns (n_assets, n_assets)
        """
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.n_assets = returns.shape[0]
        
    def optimize_mean_variance(self, 
                             target_return: Optional[float] = None,
                             risk_aversion: Optional[float] = None) -> OptimizationResult:
        """
        Perform mean-variance optimization.
        
        Args:
            target_return: Target portfolio return (for minimum variance)
            risk_aversion: Risk aversion parameter (for utility maximization)
        """
        P = cvxopt.matrix(self.cov_matrix)
        q = cvxopt.matrix(np.zeros(self.n_assets))
        
        # Constraints: sum of weights = 1
        A = cvxopt.matrix(np.ones((1, self.n_assets)))
        b = cvxopt.matrix(np.ones(1))
        
        # Non-negative constraints
        G = cvxopt.matrix(-np.eye(self.n_assets))
        h = cvxopt.matrix(np.zeros(self.n_assets))
        
        if target_return is not None:
            # Add return constraint
            A_return = cvxopt.matrix(np.vstack((
                np.ones(self.n_assets),
                np.mean(self.returns, axis=1)
            )))
            b_return = cvxopt.matrix([1.0, target_return])
            
            solution = cvxopt.solvers.qp(P, q, G, h, A_return, b_return)
        else:
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
            
        if solution['status'] == 'optimal':
            weights = np.array(solution['x']).flatten()
            obj_value = float(solution['primal objective'])
            return OptimizationResult(weights, obj_value, True, "Optimization successful")
        else:
            return OptimizationResult(
                np.zeros(self.n_assets),
                float('inf'),
                False,
                f"Optimization failed: {solution['status']}"
            )
    
    def optimize_risk_parity(self) -> OptimizationResult:
        """Optimize portfolio using risk parity strategy."""
        
        def risk_parity_objective(weights):
            weights = np.array(weights).reshape(-1)
            port_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
            risk_contrib = weights * (self.cov_matrix @ weights) / port_vol
            return np.sum((risk_contrib - risk_contrib.mean())**2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Sum to 1
            {'type': 'ineq', 'fun': lambda x: x}  # Non-negative
        ]
        
        x0 = np.ones(self.n_assets) / self.n_assets
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )
        
        return OptimizationResult(
            result.x,
            result.fun,
            result.success,
            result.message
        )
    
    def optimize_maximum_diversification(self) -> OptimizationResult:
        """Optimize portfolio using maximum diversification strategy."""
        
        def diversification_ratio(weights):
            weights = np.array(weights).reshape(-1)
            port_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
            weighted_vols = weights * np.sqrt(np.diag(self.cov_matrix))
            return -(np.sum(weighted_vols) / port_vol)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        
        x0 = np.ones(self.n_assets) / self.n_assets
        result = minimize(
            diversification_ratio,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'ftol': 1e-12}
        )
        
        return OptimizationResult(
            result.x,
            -result.fun,  # Convert back to positive DR
            result.success,
            result.message
        )
    
    def optimize_minimum_correlation(self) -> OptimizationResult:
        """Optimize portfolio using minimum correlation strategy."""
        
        # Convert covariance to correlation
        std_dev = np.sqrt(np.diag(self.cov_matrix))
        corr_matrix = self.cov_matrix / np.outer(std_dev, std_dev)
        
        def correlation_objective(weights):
            weights = np.array(weights).reshape(-1)
            numerator = weights.T @ corr_matrix @ weights
            denominator = weights.T @ weights
            return numerator / denominator
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        
        x0 = np.ones(self.n_assets) / self.n_assets
        result = minimize(
            correlation_objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'ftol': 1e-12}
        )
        
        return OptimizationResult(
            result.x,
            result.fun,
            result.success,
            result.message
        )
