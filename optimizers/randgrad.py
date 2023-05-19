from qiskit.algorithms.optimizers import SPSA
import numpy as np

class RandGrad(SPSA):
    """
    Overriding qiskit SPSA by changing the sampling of grad by selecting random point on a sphere. 
    """
    def _point_estimate(self, loss, x, eps, num_samples):
        """The gradient estimate at point x."""
        # set up variables to store averages
        value_estimate = 0
        gradient_estimate = np.zeros(x.size)

        samples = []
        def sample_sphere(size):
            mu = np.random.randn(x.size)
            mu /= np.linalg.norm(mu) 
            return mu * np.sqrt(x.size)

        # iterate over the directions
        deltas1 = [
            sample_sphere(x.size) for _ in range(num_samples)
        ]
        deltas2 = None

        for i in range(num_samples):
            delta1 = deltas1[i]
            delta2 = deltas2[i] if self.second_order else None

            value_sample, gradient_sample, hessian_sample = self._point_sample(
                loss, x, eps, delta1, delta2
            )
            value_estimate += value_sample
            gradient_estimate += gradient_sample

            if self.second_order:
                hessian_estimate += hessian_sample

        return (
            value_estimate / num_samples,
            gradient_estimate / num_samples,
            None,
        )
