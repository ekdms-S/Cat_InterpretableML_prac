from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class Regressor:
    def ElasticNet(self):
        return ElasticNet(alpha=0.01)

    def GaussianProcess(self):
        kernel = 1.0 * Matern(length_scale=1.0, nu=2.5)
        return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30)

    def __call__(self, name):
        if name == 'ELN':
            regressor = self.ElasticNet()
        elif name == 'GPR':
            regressor = self.GaussianProcess()
        else:
            raise Exception

        return regressor