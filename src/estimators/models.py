import torch
import torch.nn as nn
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from skorch import NeuralNetRegressor
from torch import optim
from sklearn.ensemble import GradientBoostingClassifier
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.functions import StepFunction
from sksurv.linear_model import IPCRidge

class KM:
    def __init__(self):
        return
    
    def fit(self,X,y):
        x,y = kaplan_meier_estimator(y["e"], y["t"])
        self.model = StepFunction(x,y)
    
    def predict(self,X):
        return self.model(X)
    
    def predict_survival_function(self,X, return_array = False):
        if return_array:
            return self.model.y[None,:].repeat(len(X),0)
        else:
            return [self.model for _ in range(len(X))]
    
class Model: 
    
    def __init__(self, model_name='LogisticRegression', hp={}, model_type='binary'): 
        self.model_name = model_name
        if model_name == 'MLP': 
            if torch.cuda.is_available():
                self.device = torch.device('cuda:1')
            else:
                self.device  = torch.device('cpu')
        self.model = self._init_model(model_name, hp)
        self.model_type = model_type
        
    
    def _init_model(self, name, params): 
        if name == 'LogisticRegression':
            C = params.get('C',1.)
            return Pipeline([('var_threshold', VarianceThreshold()), \
                             ('LR', LogisticRegression(C=C, max_iter = 1))])
        elif name == 'GradientBoostingClassifier':
            lr = params.get('learning_rate', 1.)
            ne = params.get('n_estimators', 1.)
            md = params.get('max_depth', 1.)
            msl = params.get('min_samples_leaf', 1.)
            mss = params.get('min_samples_split', 1.)
            mf = params.get('max_features', 1.)
            rs = int(params.get('random_state', 1.))
            return GradientBoostingClassifier(learning_rate = lr, n_estimators = ne,\
                                              max_depth = md, min_samples_leaf = msl,\
                                              min_samples_split = mss, max_features = mf,\
                                              random_state = rs)
        elif name == 'RandomForestRegressor': 
            nt = params.get('n_estimators',50)
            md = params.get('max_depth', 5)
            mn = params.get('min_samples_split', 10)
            mf = params.get('max_features', 'all')
            return RandomForestRegressor(n_estimators=nt, max_depth=md, \
                                          min_samples_split=mn, max_features=mf)         
        elif name == 'Lasso': 
            alpha = params.get('alpha',1.)
            return Lasso(alpha=alpha, max_iter=1000)
        elif name == 'LinearRegression': 
            return LinearRegression()
        elif name == 'MLP': 
            hidden_layer_sizes = params.get('hidden_layer_sizes', (50,50))
            activation = params.get('activation', 'relu')
            solver = params.get('solver', 'adam')
            alpha = params.get('alpha', .001)
            learning_rate = params.get('learning_rate', 'adaptive')
            learning_rate_init = params.get('learning_rate_init', 1e-3)
            max_iter = params.get('max_iter', 200)
            input_dim = params.get('input_dim',-1)
            
            if solver == 'adam': 
                o = optim.Adam
            elif solver == 'sgd': 
                o = optim.SGD
                
#             return NeuralNetRegressor(module=MLP,
#                                module__hidden_layer_sizes=hidden_layer_sizes,
#                                module__activation=activation, 
#                                module__input_dim=input_dim,
#                                optimizer=o,
#                                optimizer__lr=learning_rate_init,
#                                optimizer__weight_decay=alpha,
#                                max_epochs=max_iter,
#                                device=self.device,
#                                train_split=None,
#                                verbose=1
#                               ) 
            return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation,
                                solver=solver,
                                alpha=alpha,
                                learning_rate=learning_rate,
                                learning_rate_init=learning_rate_init,
                                max_iter=max_iter)
        
        elif name == "CoxModel":
            alpha = params.get('alpha',0.)
            return CoxPHSurvivalAnalysis(alpha=alpha)
        
        elif name == "AFTModel":
            alpha = params.get('alpha',0.)
            return IPCRidge(alpha = alpha) 
        
        elif name == "KM":
            return KM()

    def fit(self, X, y): 
#         if self.model_name == 'MLP': 
#             X = X.astype(np.float32)
#             y = y.reshape(-1,1).astype(np.float32)

        try:
            self.model.fit(X,y)
        except ValueError as e:
            print(e)
            breakpoint()
    
    def predict(self, X, return_array = True): 
        if self.model_type == 'binary':
            return self.model.predict_proba(X)[:,1]
        elif self.model_type == 'survival':
            if isinstance(self.model,IPCRidge):
                return self.model.predict(X)
            else:
                return self.model.predict_survival_function(X, return_array= return_array) # the survival function.
        else:
            return self.model.predict(X).squeeze()
    
    def compute_metric(self, y_true, y_predict): 
        if self.model_type == 'binary': 
            return roc_auc_score(y_true, y_predict)
        elif self.model_type == 'continuous': 
            return -mean_squared_error(y_true, y_predict)
        elif self.model_type == 'survival':
            if len(y_predict.shape)==1:
                y_predict = y_predict[:,None]
            return concordance_index_censored(y_true["e"], y_true["t"], 1-y_predict[:,0])
        elif self.model_type == 'KM':
            return 0
        else: 
            raise ValueError('metric can only be computed for binary and continuous outcomes')

class OracleModel(Model): 

    def __init__(self, model_name='ResponseSurfaceOracle-1', hp={}, model_type='continuous', params={}):
        self.params = params
        self.model_name = model_name 
        self.model  = self._init_model()
    
    def _init_model(self): 
        if 'ResponseSurface' in self.model_name: 
            beta_B = self.params['oracle_params']['response_surface']['beta_B']
            gamma  = self.params['oracle_params']['response_surface']['gamma']
            omega  = self.params['oracle_params']['response_surface']['omega']
            W      = self.params['oracle_params']['response_surface']['W']
            return (beta_B, gamma, omega, W)
        elif 'SelectionModel' in self.model_name: 
            P_S0 = self.params['oracle_params']['selection_model']['P_S0']
            P_X_S0  = self.params['oracle_params']['selection_model']['P_X_S0']
            P_X_S1  = self.params['oracle_params']['selection_model']['P_X_S1']
            prob_indices_obs      = self.params['oracle_params']['selection_model']['prob_indices_obs']
            prob_indices_rct      = self.params['oracle_params']['selection_model']['prob_indices_rct']
            return (P_S0, P_X_S0, P_X_S1, prob_indices_obs, prob_indices_rct)
    
    def predict(self, X, orig_idx_test=[]):         
        if self.model_name == 'ResponseSurfaceOracle-0': 
            beta_B, gamma, _, W = self.model 
            num_covariates = W.shape[0]
            X_orig = X[...,:num_covariates]
            Z = X[...,num_covariates:]
            if self.params['response_surface']['ctr'] == 'linear':     
                return (np.matmul((X_orig+W[None,:]),beta_B) \
                    + np.matmul(Z,gamma)).squeeze()
            elif self.params['response_surface']['ctr'] == 'non_linear': 
                return (np.exp(np.matmul((X_orig+W[None,:]),beta_B)) \
                    + np.matmul(Z,gamma)).squeeze()
        elif self.model_name == 'ResponseSurfaceOracle-1': 
            beta_B, gamma, omega, W = self.model 
            num_covariates = W.shape[0]
            X_orig = X[...,:num_covariates]
            Z = X[...,num_covariates:]
            breakpoint()
            return (np.matmul(X_orig,beta_B) - omega \
                    + np.matmul(Z,gamma)).squeeze()
        elif self.model_name == 'SelectionModelOracle': 
            P_S0, P_X_S0, P_X_S1, prob_indices_obs, prob_indices_rct = self.model # P_X_S1 (wi / w_bar)
            prob_indices = np.array(list(prob_indices_obs) + list(prob_indices_rct))
            assert len(orig_idx_test) != 0, 'need to pass in orig_idx_test for oracle selection model!'
            wi_wbar = np.zeros((X.shape[0],)) # p(X|S=1) = ret_X
            wi_wbar = P_X_S1[prob_indices[orig_idx_test]]
            wbar_wi = 1. / wi_wbar
            P_S1_X = 1 / (1 + (P_S0 / (1-P_S0))*wbar_wi)
            return P_S1_X

class MLP(nn.Module):
    def __init__(self, hidden_layer_sizes, activation, input_dim):
        super(MLP, self).__init__()
        self.num_layers = len(hidden_layer_sizes)+1
        self.layers = nn.ModuleList()
        for i in range(self.num_layers): 
            if i == 0: 
                self.layers.append(nn.Linear(input_dim,\
                                    hidden_layer_sizes[0]))
            elif i == self.num_layers-1: 
                self.layers.append(nn.Linear(hidden_layer_sizes[-1],1))
            else: 
                self.layers.append(nn.Linear(hidden_layer_sizes[i-1],\
                                    hidden_layer_sizes[i]))
        if activation == 'relu':
            self.m = nn.ReLU()
        elif activation == 'tanh':
            self.m = nn.Tanh()
        assert len(self.layers) == self.num_layers

    def forward(self, X, **kwargs):
        for i in range(self.num_layers-1): 
            X = self.layers[i](X)
            X = self.m(X)
        return self.layers[-1](X)
