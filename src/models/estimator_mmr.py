import pandas as pd 
import numpy as np
from scipy.stats import norm
import sys 

from src.models.falsifier_mmr import FalsifierMMR

# sklearn logistic regression (propensity score)
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.utils import resample
from itertools import product
from src.estimators.models import Model, OracleModel
import src.models.model_util as model_util


def predict_survival(pred_fun,Y):
    outs = []
    for i, pred_ in enumerate(pred_fun):
        if Y[i] < pred_.x[0]:
            outs.append(1.)
        elif Y[i] > pred_.x[-1]:
            outs.append(0.)
        else:
            outs.append(pred_(Y[i]))
    return np.stack(outs)

class OutcomeEstimator: 

    def __init__(self, rct_table, obs_table, params={}, rct_partial=True):
        self.params     = params 
        # stack datasets 
        stacked_table = model_util._stack(params, rct_table, obs_table)
        if self.params['ihdp'] and rct_partial:
            self.stacked_table = stacked_table[stacked_table['b.marr'] == 1.]
        else: 
            self.stacked_table = stacked_table
    
    def get_stacked_table(self): 
        return self.stacked_table
     
    def _hp_selection(self,
                      data, 
                      test_size=0.2, 
                      seed=42, 
                      model_name='LogisticRegression', 
                      hp={},
                      model_type='prop_score'): 
        
        X   = data['X']; y = data['y']        
        if model_type == 'binary': 
            tr_idxs, val_idxs = train_test_split(np.arange(X.shape[0]),\
                            test_size=test_size,random_state=seed,stratify=y)
        else: 
            tr_idxs, val_idxs = train_test_split(np.arange(X.shape[0]),\
                            test_size=test_size,random_state=seed)
        
        X_train, X_val = X[tr_idxs], X[val_idxs]
        y_train, y_val = y[tr_idxs], y[val_idxs]
        best_hp = None # to store best hps 

        param_names = hp.keys()
        param_lists = [hp[k] for k in param_names]
        print('')
        for elem in product(*param_lists): 
            print(f'[trying hp {elem} for {model_name}]')
            params = {k:elem[i] for i,k in enumerate(param_names)}
            params['input_dim'] = X_train.shape[1]
            
            model = Model(model_name, hp=params, model_type=model_type)
            model.fit(X_train, y_train)
            y_predict = model.predict(X_val)
            metric = model.compute_metric(y_val, y_predict)
            
            if best_hp is None or metric > best_hp[0]: 
                best_hp = (metric, params)

        print(f'best hp: {best_hp[1]}')
        return best_hp[1]
    

    def _create_censoring_response(self, censoring_type, Y_p, Y, C, D, idx):
        """
        Generates the X and y vector for computing the censoring distribution depending on the censoring case.
        """
        if censoring_type == "cond_indep": #return the censored values when observed
            c_vec = np.array([((1-D[idx][i]), Y_p[idx][i]) for i in
                      range(len(Y_p[idx]))], dtype=[('e', bool), ('t', float)])
        elif censoring_type == "type_1_indep_x": # return all the censored values ( we consider censoring is always observed)
            c_vec = np.array([((1), C[idx][i]) for i in
                      range(len(Y_p[idx]))], dtype=[('e', bool), ('t', float)])
        return c_vec
    

    def _compute_censoring_survival(self, train_vecs, test_vecs, train_sub_idx, test_sub_idx):
        """
        Compute the censoring survival for each data point in the test sets.

        train_sub_idx and test_sub_idx are indices to select only one part of the train and test data (for instance only the treated patients)
        censoring survival is set to 1 for the units not in train_sub_idx and test_sub_idx.

        """
        Xtrain, Yptrain, Ttrain, Strain, Dtrain, Ctrain, Ytrain = train_vecs
        Xtest, Yptest, Ttest, Stest, Dtest, Ctest, Ytest = test_vecs

        if self.params["censoring_model"]["model_name"] == "MMR":
            return np.ones(Ytest.shape[0])
        
        # RETURN 1 IF CLASSICAL MMR

        c_vec_train = self._create_censoring_response( censoring_type = self.params["censoring_type"],
                                                             Y_p =  Yptrain,
                                                             Y=  Ytrain,
                                                              C = Ctrain,
                                                               D= Dtrain,
                                                                idx =  train_sub_idx)
        
        c_vec_test = self._create_censoring_response( censoring_type = self.params["censoring_type"],
                                    Y_p =  Yptest,
                                        Y=  Ytest,
                                        C = Ctest,
                                        D= Dtest,
                                        idx =  test_sub_idx)

        data_c = {'X': Xtrain[train_sub_idx], 'y': c_vec_train}

        best_hp_c = self._hp_selection(data_c, 
                test_size=0.2, 
                seed=self.params['cross_fitting_seed'], 
                model_name=self.params['censoring_model']['model_name'], 
                hp=self.params['censoring_model']['hp'],
                model_type=self.params['censoring_model']['model_type'])
        
        c_mod = Model(self.params['censoring_model']['model_name'], 
            hp=best_hp_c, model_type=self.params['censoring_model']['model_type'])
        
        c_mod.fit(Xtrain[train_sub_idx], c_vec_train) 
        p_c_x = c_mod.predict(Xtest[test_sub_idx],return_array=False)
        p_c_s = predict_survival(p_c_x, Yptest[test_sub_idx])
        p_out = np.ones(Ytest.shape[0])
        p_out[test_sub_idx] = p_c_s

        return p_out



    def _compute_phi0_estimates(self, cross_fitting_seed=42, S_target = 0): 
        """
        Compute the phi0 signal.

        Normally, this is done for S=0. Nevertheless, you can also compute the same signal for the observational study by setting S_target = 1.
        """

        X, Y_p, T, S, Y, C, D = model_util._get_numpy_arrays(self.params, self.stacked_table)

        sub  = self.stacked_table[['S','treat']]
        sub_study = sub[sub['S'] == S_target]
        T_study = sub_study['treat'].values

        p_T1_S = np.sum(T_study) / T_study.shape[0]
        p_T0_S = np.sum(1-T_study) / T_study.shape[0]


        # basic sample splitting -- no need to do cross fitting? 
        cvk = StratifiedKFold(n_splits=3, shuffle=True, random_state=cross_fitting_seed)
        orig_idx = np.arange(S.shape[0])
        final_data = []

        for train_idx, test_idx in cvk.split(X,S):
            Xtrain, Yptrain, Ttrain, Strain, Dtrain, Ctrain, Ytrain = X[train_idx], Y_p[train_idx], T[train_idx], S[train_idx], D[train_idx], C[train_idx], Y[train_idx]
            Xtest, Yptest, Ttest, Stest, Dtest, Ctest, Ytest = X[test_idx], Y_p[test_idx], T[test_idx], S[test_idx], D[test_idx], C[test_idx], Y[test_idx]
            orig_idx_test = orig_idx[test_idx]

            #S Model
            if 'Oracle' not in self.params['selection_model']['model_name']:

                #Selection model
                print('\nHP search for selection model for RCT signal.')
                data_prop = {'X': Xtrain, 'y': Strain}
                best_hp_prop = self._hp_selection(data_prop, 
                        test_size=0.2, 
                        seed=self.params['cross_fitting_seed'], 
                        model_name=self.params['selection_model']['model_name'], 
                        hp=self.params['selection_model']['hp'],
                        model_type=self.params['selection_model']['model_type'])
                s = Model(self.params['selection_model']['model_name'], 
                    hp=best_hp_prop, model_type=self.params['selection_model']['model_type'])
                s.fit(Xtrain, Strain) 
                p_s1_x = s.predict(Xtest)

                # Censoring Model with A=1
                c_idx_train = (Ttrain == 1) * (Strain == S_target)
                c_idx_test = (Ttest == 1) * (Stest == S_target)

                p_c_1 = self._compute_censoring_survival(train_vecs = (Xtrain, Yptrain, Ttrain, Strain, Dtrain, Ctrain, Ytrain),
                                                test_vecs = (Xtest, Yptest, Ttest, Stest, Dtest, Ctest, Ytest),
                                                train_sub_idx = c_idx_train,
                                                test_sub_idx = c_idx_test)
                
                # Censoring model with A=0
                c_idx_train = (Ttrain == 0) * (Strain == S_target)
                c_idx_test = (Ttest == 0) * (Stest == S_target)
                p_c_0 = self._compute_censoring_survival(train_vecs = (Xtrain, Yptrain, Ttrain, Strain, Dtrain, Ctrain, Ytrain),
                                                test_vecs = (Xtest, Yptest, Ttest, Stest, Dtest, Ctest, Ytest),
                                                train_sub_idx = c_idx_train,
                                                test_sub_idx = c_idx_test)
    
            else: 
                print('\nOracle selection model for RCT signal.')
                raise("Not implemented with censoring yet !")
                s = OracleModel(self.params['selection_model']['model_name'], \
                    hp={}, model_type='continuous', params=self.params)
                p_s1_x = s.predict(Xtest, orig_idx_test)

            p_s_x = (1-p_s1_x) * (1-S_target) + p_s1_x * S_target # P(S==S_target|X)

            U1_test = ((1-Stest) * (Dtest))/(1-p_s_x) * ((Ttest*Yptest) / (p_T1_S* p_c_1))
            U0_test = ((1-Stest) * (Dtest))/(1-p_s_x) * (((1-Ttest)*Yptest) / (p_T0_S* p_c_0))
            # final signals
            #U1_test = ((1-Stest)/(1-p_s1_x)) * (  (Ttest*Ytest) / p_T1_S0  ) 
            #U0_test = ((1-Stest)/(1-p_s1_x)) * (  ((1-Ttest)*Ytest) / p_T0_S0  ) 

            final_data.append((orig_idx_test, U1_test[:,None], U0_test[:,None]))
            
        # final signals
        U1_final = np.concatenate([elem[1] for elem in final_data], axis=0)
        U0_final = np.concatenate([elem[2] for elem in final_data], axis=0)
        orig_idxs_shuffled = np.concatenate([elem[0] for elem in final_data], axis=0)
        return (U1_final[orig_idxs_shuffled.argsort()].squeeze(), \
            U0_final[orig_idxs_shuffled.argsort()].squeeze())
        # U1 = ((1-S)/p_S0) * (  (T*Y) / p_T1_S0  ) 
        # U0 = ((1-S)/p_S0) * (  ((1-T)*Y) / p_T0_S0  ) 
        # return (U1, U0)
    
    def _compute_obs_estimates(self, cross_fitting_seed=42): 
        cvk = StratifiedKFold(n_splits=3, shuffle=True, random_state=cross_fitting_seed)
        X, Y, T, S, Yt, Ct, D = model_util._get_numpy_arrays(self.params, self.stacked_table)

        orig_idx = np.arange(S.shape[0])

        final_data = []

        for train_idx, test_idx in cvk.split(X,S): 
            Xtrain, Ytrain, Ttrain, Strain = X[train_idx], Y[train_idx], T[train_idx], S[train_idx]
            Xtest, Ytest, Ttest, Stest = X[test_idx], Y[test_idx], T[test_idx], S[test_idx] 
            orig_idx_train = orig_idx[train_idx]; orig_idx_test = orig_idx[test_idx]

            # stratifying data by RCT and obs
            source_idxs = np.where(Strain == 1)
            Xtrain_obs, Ytrain_obs, Ttrain_obs = Xtrain[source_idxs], Ytrain[source_idxs], Ttrain[source_idxs]
            source_idxs = np.where(Stest == 1)
            Xtest_obs, Ytest_obs, Ttest_obs = Xtest[source_idxs], Ytest[source_idxs], Ttest[source_idxs]

            # propensity score model 
            print('\nHP search for propensity model')
            data_prop = {'X': Xtrain_obs, 'y': Ttrain_obs}
            best_hp_prop = self._hp_selection(data_prop, 
                      test_size=0.2, 
                      seed=self.params['cross_fitting_seed'], 
                      model_name=self.params['propensity_model']['model_name'], 
                      hp=self.params['propensity_model']['hp'],
                      model_type=self.params['propensity_model']['model_type'])
            g = Model(self.params['propensity_model']['model_name'], 
                hp=best_hp_prop, model_type=self.params['propensity_model']['model_type'])
            g.fit(Xtrain_obs, Ttrain_obs)

            # selection model, P(S=1|X)
            # marr_number = np.where(self.stacked_table.columns.values == 'b.marr')[0][0]-3
            # pdb.set_trace()
            # overlap_idxs = np.where(Xtrain[:,marr_number] == 1)
            # Xtrain_ov = Xtrain[overlap_idxs]
            # Strain_ov = Strain[overlap_idxs]
            if 'Oracle' not in self.params['selection_model']['model_name']: 
                print('\nHP search for selection model')
                data_prop = {'X': Xtrain, 'y': Strain}
                best_hp_prop = self._hp_selection(data_prop, 
                        test_size=0.2, 
                        seed=self.params['cross_fitting_seed'], 
                        model_name=self.params['selection_model']['model_name'], 
                        hp=self.params['selection_model']['hp'],
                        model_type=self.params['selection_model']['model_type'])
                s = Model(self.params['selection_model']['model_name'], 
                    hp=best_hp_prop, model_type=self.params['selection_model']['model_type'])
                s.fit(Xtrain, Strain)
            else:
                print('\nOracle selection model for OBS signal.') 
                s = OracleModel(self.params['selection_model']['model_name'], hp={}, model_type='continuous', params=self.params)

            # response surface model, P(Y|X,T,S=1)
            Y1train_obs = Ytrain_obs[Ttrain_obs == 1]
            Y0train_obs = Ytrain_obs[Ttrain_obs == 0]
            X1train_obs = Xtrain_obs[Ttrain_obs == 1, :]
            X0train_obs = Xtrain_obs[Ttrain_obs == 0, :]

            ## response surface model (T-learner)
            # XTtrain_obs = np.concatenate((Xtrain_obs,Ttrain_obs[:,None]),axis=1)
            # data_resp = {'X': XTtrain_obs, 'y': Ytrain_obs}
            data_resp1 = {'X': X1train_obs , 'y': Y1train_obs}
            data_resp0 = {'X': X0train_obs , 'y': Y0train_obs}
            if len(self.params['response_surface_1']['hp'].keys()) != 0:  # Obsolete?
                print('\nHP search for response surface model')
                best_hp_resp1 = self._hp_selection(data_resp1, 
                      test_size=0.2, 
                      seed=self.params['cross_fitting_seed'],
                      model_name=self.params['response_surface_1']['model_name'], 
                      hp = self.params['response_surface_1']['hp'],
                      model_type=self.params['response_surface_1']['model_type'])
                best_hp_resp1['input_dim'] = X1train_obs.shape[1]
            else: 
                best_hp_resp1 = {}
                print(f'No hp selection for response surface 1. Fitting with default values.')
            
            if len(self.params['response_surface_0']['hp'].keys()) != 0:  # Obsolete?
                print('\nHP search for response surface model')
                best_hp_resp0 = self._hp_selection(data_resp0, 
                      test_size=0.2, 
                      seed=self.params['cross_fitting_seed'], 
                      model_name=self.params['response_surface_0']['model_name'], 
                      hp = self.params['response_surface_0']['hp'],
                      model_type=self.params['response_surface_0']['model_type'])
                best_hp_resp0['input_dim'] = X0train_obs.shape[1]
            else: 
                best_hp_resp0 = {}
                print(f'No hp selection for response surface 0. Fitting with default values.')
            
            if 'Oracle' not in self.params['response_surface_1']['model_name']: 
                f1  = Model(self.params['response_surface_1']['model_name'], hp=best_hp_resp1, model_type=self.params['response_surface_1']['model_type'])
                f0  = Model(self.params['response_surface_0']['model_name'], hp=best_hp_resp0, model_type=self.params['response_surface_0']['model_type'])
                f1.fit(X1train_obs, Y1train_obs)
                f0.fit(X0train_obs, Y0train_obs)
            else: 
                f1  = OracleModel(self.params['response_surface_1']['model_name'], hp={}, model_type='continuous', params=self.params)
                f0  = OracleModel(self.params['response_surface_0']['model_name'], hp={}, model_type='continuous', params=self.params)
            
            Xt_pred_f1 = f1.predict(Xtest)
            Xt_pred_f0 = f0.predict(Xtest)

            X_pred_g1 = g.predict(Xtest) # prop score
            X_pred_g0 = 1-X_pred_g1
            if 'Oracle' not in self.params['selection_model']['model_name']: 
                X_pred_s = s.predict(Xtest) # p(S=1|X)
            else: 
                X_pred_s = s.predict(Xtest, orig_idx_test)
            
            pS_0 = np.sum(1-Stest) / Stest.shape[0]
            
            # Ut_test1
            ipw_signal = ((1-X_pred_s)/X_pred_s)*(Ttest*(Ytest - Xt_pred_f1) / X_pred_g1)
            rs_signal  = Xt_pred_f1 
            # Ut_test1 = (1/pS_0)*((1-Stest)*rs_signal + Stest*ipw_signal)
            Ut_test1 = (1/(1-X_pred_s))*((1-Stest)*rs_signal + Stest*ipw_signal)

            # Ut_test0
            ipw_signal = ((1-X_pred_s)/X_pred_s)*((1-Ttest)*(Ytest - Xt_pred_f0) / X_pred_g0)
            rs_signal  = Xt_pred_f0 
            # Ut_test0   = (1/pS_0)*((1-Stest)*rs_signal + Stest*ipw_signal)
            Ut_test0   = (1/(1-X_pred_s))*((1-Stest)*rs_signal + Stest*ipw_signal)

            final_data.append((orig_idx_test, Ut_test1[:,None], Ut_test0[:,None]))
        
        print(f'number of tuples: {len(final_data)}')
        print(f'shapes of tuple 1: {[elem.shape for elem in final_data[0]]}')
        U1_obs_final = np.concatenate([elem[1] for elem in final_data], axis=0)
        U0_obs_final = np.concatenate([elem[2] for elem in final_data], axis=0)
        orig_idxs_shuffled = np.concatenate([elem[0] for elem in final_data], axis=0)
        return (U1_obs_final[orig_idxs_shuffled.argsort()].squeeze(), \
            U0_obs_final[orig_idxs_shuffled.argsort()].squeeze())
    
    def estimate_signals(self, S):
        if S == 0: 
            return self._compute_phi0_estimates()
        elif S == 1: 
            if self.params["phi1_as_phi0"]:
                return self._compute_phi0_estimates(S_target=1) #we use the same estimator for phi1 than phi0
            else:
                return self._compute_obs_estimates() # doubly robust estimator 
        
        