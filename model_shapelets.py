import numpy as np
from Lens_Modeling_Auto.auto_modeling_functions import get_kwarg_names
from copy import deepcopy
import os
import time


# SHAPELETS_indices = [i for i,x in enumerate(deepcopy(source_model_list)*len(kwargs_data)) if x == 'SHAPELETS']

# for j in SHAPELETS_indices:
# #     source_params_update[0][j] = {'n_max': n_max, 
# #                                   'center_x': kwargs_source_init_update[j]['center_x'], 
# #                                   'center_y': kwargs_source_init_update[j]['center_y'], 
# #                                   'beta': kwargs_source_init_update[j]['beta']}
    
#     beta = source_params_update[0][j]['beta']    
#     source_params_update[1][j] = {'center_x': 0.01, 'center_y': 0.01, 'beta': 0.05}    
#     source_params_update[2][j] = {'n_max': n_max}    
#     source_params_update[3][j] = {'center_x': -1.5, 'center_y': -0.5, 'beta': beta / np.sqrt(n_max + 1)}
#     source_params_update[4][j] = {'center_x': 1.5, 'center_y': 1.5, 'beta': beta * np.sqrt(n_max + 1)}
    

kwargs_params = {'lens_model': lens_params_update,
                        'source_model': source_params_update,
                        'lens_light_model': lens_light_params_update}

exec(open('Lens_Modeling_Auto/Full_Sampling.py').read())

new_results_path = results_path + '/shapelets_nmax_{}'.format(n_max) + '_image_{}'.format(it + 1)
if not exists(new_results_path):
    os.mkdir(new_results_path)

if not exists(new_results_path + '/modelPlot_results'):
    os.mkdir(new_results_path + '/modelPlot_results')
if not exists(new_results_path + '/chainPlot_results'):
    os.mkdir(new_results_path + '/chainPlot_results')
if not exists(new_results_path + '/cornerPlot_results'):
    os.mkdir(new_results_path + '/cornerPlot_results')

modelPlot_path = new_results_path + '/modelPlot_results'
chainPlot_path = new_results_path + '/chainPlot_results'
cornerPlot_path = new_results_path + '/cornerPlot_results'
#csv_path = new_results_path
exec(open('Lens_Modeling_Auto/plot_results.py').read())
exec(open('Lens_Modeling_Auto/save_to_csv_full.py').read())