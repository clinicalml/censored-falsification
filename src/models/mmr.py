from src.models.falsifier_mmr import FalsifierMMR
from src.models.estimator_mmr import OutcomeEstimator
import numpy as np
import src.models.model_util as model_util

class MMR:
    def __init__(self, oracle_params,seed, **model_cfg):
        self.model_cfg = model_cfg
        self.model_cfg['oracle_params'] = oracle_params
        self.seed = seed
    
    def run(self,data_table,
            data_dicts, 
            alpha, 
            falsification_type, 
            visualize=False,
            iter_=0):

        stacked_tables   = []
        mmr_test_signals = []
        results_to_append = []
        for _, obs_table in enumerate(data_dicts['obs']): 
            n1 = obs_table.shape[0]; n0 = data_dicts['rct-full'].shape[0]
            self.model_cfg['oracle_params']['selection_model']['P_S0'] = n0 / (n0+n1)
            oe = OutcomeEstimator(rct_table=data_dicts['rct-full'], # TODO: make this rct-full? 
                                    obs_table=obs_table,
                                    params=self.model_cfg,
                                    rct_partial=False)
            stacked_tables.append(oe.get_stacked_table())
            U_obs_a1, U_obs_a0 = oe.estimate_signals(S=1)

            '''
                Part 3: estimation of signals for RCT studies
            '''            
            U_rct_a1, U_rct_a0 = oe.estimate_signals(S=0)
            psi1 = U_obs_a1 - U_rct_a1 
            psi0 = U_obs_a0 - U_rct_a0  
            mmr_test_signals.append((psi0,psi1))
            

        '''
            Part 4: write new falsifier that incorporates MMR test 
            
        '''
        falsifier = FalsifierMMR(params=self.model_cfg, alpha=alpha,\
            kernel=self.model_cfg['kernel'], seed = self.seed, falsification_type=falsification_type)
        for k, _ in enumerate(data_dicts['obs']):  
            p_val = falsifier.run_test(stacked_tables[k], mmr_test_signals[k], B=100)
            results_add = {'iter': iter_}
            results_add['obs_study_num'] = k+1 
            results_add['obs_study_size'] = self.model_cfg['obs_dict']['sizes'][k]
            results_add['p_val'] = p_val
            results_add['reject'] = int(p_val < alpha)
            if visualize: 
                print('[Visualizing witness function!]')
                covariate_names = ['nnhealth', 'booze']; covariate_types = ['continuous', 'binary']
                f, Xmean_rep, covariate_idxs = falsifier.visualize_witness_func(stacked_tables[k], mmr_test_signals[k], \
                    covariate_names=covariate_names, covariate_types=covariate_types)
                if len(covariate_names) == 2 and 'binary' in covariate_types:
                    # precondition: always put binary covariate second
                    pos_idxs = np.where(Xmean_rep[:,covariate_idxs[1]] == 1.)
                    neg_idxs = np.where(Xmean_rep[:,covariate_idxs[1]] == 0.)
                    x_coord  = Xmean_rep[:,covariate_idxs[0]]
                    cov_mean, cov_std = data_table.get_normalizing_factors(covariate_names[0])
                    x_coord_orig = (x_coord*cov_std)+cov_mean
                    x_coord1 = x_coord_orig[pos_idxs]; x_coord0 = x_coord_orig[neg_idxs]
                    f_coord1 = f[pos_idxs]; f_coord0 = f[neg_idxs]
                    results_add['f_coord_pos'] = f_coord1; results_add['f_coord_neg'] = f_coord0 
                    results_add['x_coord_pos'] = x_coord1; results_add['x_coord_neg'] = x_coord0
                elif len(covariate_names) == 2: 
                    x1_coord = Xmean_rep[:,covariate_idxs[0]]
                    x2_coord = Xmean_rep[:,covariate_idxs[1]]
                    cov_mean1, cov_std1 = data_table.get_normalizing_factors(covariate_names[0])
                    cov_mean2, cov_std2 = data_table.get_normalizing_factors(covariate_names[1])
                    x1_coord_orig = (x1_coord*cov_std1)+cov_mean1
                    x2_coord_orig = (x2_coord*cov_std2)+cov_mean2
                    results_add['f'] = f
                    results_add['x1_coord'] = x1_coord_orig
                    results_add['x2_coord'] = x2_coord_orig
                elif len(covariate_names) == 1: 
                    x1_coord = Xmean_rep[:,covariate_idxs[0]]
                    cov_mean1, cov_std1 = data_table.get_normalizing_factors(covariate_names[0])
                    x1_coord_orig = (x1_coord*cov_std1)+cov_mean1
                    results_add['f'] = f
                    results_add['x1_coord'] = x1_coord_orig
                results_add['covariate_names'] = covariate_names
            results_to_append.append(results_add) 
        
        return results_to_append
        


    