import pandas as pd
import numpy as np

def _stack(params, rct_table, obs_table): 
    ''' 
        Stack the tables (and adjust number of covariates)
    '''
    rct_table = rct_table.copy()
    num_covariates = params['num_continuous'] + params['num_binary']
    rct_table.insert(loc=0, column=f'S', value=np.zeros((rct_table['treat'].shape[0],)))
    obs_table.insert(loc=0, column=f'S', value=np.ones((obs_table['treat'].shape[0],)))
    rct_table.drop(columns=['y1_rct','y0_rct','c1_rct','c0_rct'], inplace=True)
    obs_table.drop(columns=['y1_obs','y0_obs','c1_obs','c0_obs'], inplace=True)
    rct_rename = {'y_rct': 'y_hat', 'c_rct':'c_hat'}
    rct_rename.update({f'xprime_rct{i+1}':f'xprime{i+1}' \
                                        for i in range(num_covariates)})
    obs_rename = {'y_obs': 'y_hat', 'c_obs':'c_hat'}
    obs_rename.update({f'xprime_obs{i+1}':f'xprime{i+1}' \
                                        for i in range(num_covariates)})
    rct_table.rename(columns=rct_rename, inplace=True)
    obs_table.rename(columns=obs_rename, inplace=True)
    
    pooled_table = pd.concat((obs_table, rct_table),axis=0,sort=False).reset_index(drop=True)
    pooled_table = pooled_table.dropna(axis=1)
    return pooled_table


def _process_data_mmr(model_type, X, Y_p, T, S, Y, C, D):
        """
        If we want to process the censored data with the classical MMR, we have different strategies:
        - we drop the censored observations
        - we consider censored observations are actually observed
        """
        if model_type == "drop_censored":
            idx = D==1
            return X[idx], Y_p[idx], T[idx], S[idx], Y[idx], C[idx], D[idx]
        if model_type == "censoring_impute":
            return X, Y_p, T, S, Y, C, np.ones(D.shape) 

        return X, Y_p, T, S, Y, C, D

def _get_numpy_arrays(params, table): 

        X = table.drop(columns=['y_hat','S','treat','c_hat'], inplace=False).values
        Y = table['y_hat'].values
        C = table['c_hat'].values
        T = table['treat'].values
        S = table['S'].values
        D = (table["c_hat"] >= table["y_hat"]).astype(int).values
        Y_p = D*Y + (1-D)*C # we only return the first of both values.
        
        if params["censoring_model"]["model_name"] == "MMR":
              return _process_data_mmr(model_type = params["censoring_model"]["model_type"],
                                       X= X,
                                        Y_p =  Y_p,
                                        T = T,
                                         S = S,
                                          Y = Y, C = C, D =D )

        return X, Y_p, T, S, Y, C, D