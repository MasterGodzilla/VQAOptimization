from qiskit.algorithms.optimizers import SPSA
import numpy as np

class SPSAMomentum(SPSA):
    """
    SPSAMomentum overrides the original SPSA optimizer. 

    new parameters:
        mt (np.ndarray): exponential average of previous gradients
        current_iter (int): for normalizing the mt
        beta (float = 0.9): decay constant
        
    
    overrode function:
        _compute_update(self, loss, x, k, eps, lse_solver)
    
    For more information about the original implementation of SPSA, see
    'https://qiskit.org/documentation/stable/0.42/_modules/qiskit/algorithms/optimizers/spsa.html#SPSA'
    """
    mt = None
    current_iter = 0
    beta = 0.9

    def _compute_update(self, loss, x, k, eps, lse_solver):
        """
        :math: 'm_t = (\beta m_{t-1} + (1-\beta) g)'
        :math: '\hat m_t = m_t/(1 - \beta^t)'
        """
        value, grad = super()._compute_update(loss, x, k, eps, lse_solver)
        if self.current_iter == 0: # if clause avoids initializing mt without dim
            self.mt = grad * (1 - self.beta)
        else :
            self.mt = (self.beta * self.mt + (1-self.beta) * grad) 
        self.current_iter += 1
        #print ("current iter:", self.current_iter)
        #print (self.mt/(1-self.beta**self.current_iter))
        return value, self.mt/(1-self.beta**self.current_iter)

class SPSAAdam(SPSA):
    """
    SPSAAdam overrides the original SPSA optimizer.

    new parameters:
        mt (np.ndarray): exponential average of previous gradients
        vt (np.ndarray6): exponential average of square of gradients
        current_iter (int): for normalizing the mt
        beta1 (float = 0.9): decay constant
        beta2 (float = 0.999): decay constant for second order info
        adam_eps (float = 1e-8): to avoid numerical error

    overrode function:
        _compute_update(self, loss, x, k, eps, lse_solver)

    Reference:
        [1] 'https://qiskit.org/documentation/stable/0.42/_modules/qiskit/algorithms/optimizers/spsa.html#SPSA'
        [2] 'https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc'
    """
    mt = None
    vt = 0
    current_iter = 0
    beta1 = 0.9
    beta2 = 0.999
    adam_eps = 1e-8

    def _compute_update(self, loss, x, k, eps, lse_solver):
        """
        """
        value, grad = super()._compute_update(loss, x, k, eps, lse_solver)
        if self.current_iter == 0:
            self.mt = grad * (1 - self.beta1)
        else :
            self.mt = (self.beta1 * self.mt + (1-self.beta1) * grad)
        self.vt = self.beta2*self.vt + (1 - self.beta2) * np.sum(np.square(grad))
        self.current_iter += 1
        mhat = self.mt / (1 - self.beta1**self.current_iter)
        vhat = self.vt / (1- self.beta2 ** self.current_iter)
        what = mhat / (np.sqrt(vhat) + self.adam_eps)
        #print ("current iter:", self.current_iter)
        #print (what)
        return value, what
