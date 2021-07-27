import numpy as np
from copy import deepcopy

#source_model_list.append('SHAPELETS')
kwargs_init = deepcopy(kwargs_result)
kwargs_source_init_update = []
kwargs_source_sigma_update = []
kwargs_source_fixed_update = []
kwargs_source_lower_update = []
kwargs_source_upper_update = []

for x in kwargs_init['kwargs_source']:
    kwargs_source_init_update.append(x)
    kwargs_source_init_update.append({'n_max': n_max, 
                                      'center_x': x['center_x'], 
                                      'center_y': x['center_y'], 
                                      'beta': x['R_sersic']})

for x in source_params[1]:
    kwargs_source_sigma_update.append(deepcopy(x))
    kwargs_source_sigma_update.append({})

for x in source_params[2]:
    kwargs_source_fixed_update.append(deepcopy(x))
    kwargs_source_fixed_update.append({})

for x in source_params[3]:
    kwargs_source_lower_update.append(deepcopy(x))
    kwargs_source_lower_update.append({})

for x in source_params[4]:
    kwargs_source_upper_update.append(deepcopy(x))
    kwargs_source_upper_update.append({})
    
    
source_params_update = [deepcopy(kwargs_source_init_update),
                        deepcopy(kwargs_source_sigma_update),
                        deepcopy(kwargs_source_fixed_update),
                        deepcopy(kwargs_source_lower_update),
                        deepcopy(kwargs_source_upper_update)]

kwargs_fixed['kwargs_source'] = kwargs_source_fixed_update
