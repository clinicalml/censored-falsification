## TODO: merge this file with whi_experiment.py; lots of overlapping code 

from distutils.util import strtobool
from turtle import pos
import pandas as pd 
import numpy as np
from hydra.utils import instantiate
import hydra
from omegaconf import OmegaConf
import pprint

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg):

    from numpy.random import default_rng
    rng = default_rng(cfg['seed'])

    confounder_seeds = rng.choice(range(1000), size=(cfg.experiment.num_iters,))
    noise_seeds = rng.choice(range(1000), size=(cfg.experiment.num_iters,))

    results = []
    for iter_ in range(cfg.experiment.num_iters): 
        print(f'Simulation Number {iter_+1}')
        #params['confounder_seed'] = confounder_seeds[iter_]
        #params['noise_seed'] = noise_seeds[iter_]

        ''' 
            Part 1: data simulation piece 
        ''' 
        
        print(f'data generation parameters:')
        #oracle_params = OmegaConf.to_container(cfg.oracle, resolve=True)
        data_cls = instantiate(cfg.data)(confounder_seed=confounder_seeds[iter_], noise_seed=noise_seeds[iter_])
        data_cls.generate_dataset()

        data_dicts = data_cls.get_datasets()
        
        model_cls = instantiate(cfg.model)(seed = cfg["seed"],oracle_params = data_cls.oracle_params)
        
        results = model_cls.run(data_cls, data_dicts, alpha = cfg.experiment.alpha, iter_=iter_, falsification_type=cfg.experiment.falsification_type)
        pprint.pprint(results, sort_dicts=False)
    return results     

if __name__ == '__main__': 
    main()
    