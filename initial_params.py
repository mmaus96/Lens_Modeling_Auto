# lens 
fixed_lens = []
kwargs_lens_init = []
kwargs_lens_sigma = []
kwargs_lower_lens = []
kwargs_upper_lens = []

fixed_lens.append({})  # for this example, we fix the power-law index of the lens model to be isothermal
kwargs_lens_init.append({'theta_E': 1.2, 'e1': 0., 'e2': 0.,
                         'center_x': 0., 'center_y': 0.})
kwargs_lens_sigma.append({'theta_E': .3, 'e1': 0.5, 'e2': 0.5,
                         'center_x': 0.1, 'center_y': 0.1})
kwargs_lower_lens.append({'theta_E': 0.1, 'e1': -1, 'e2': -1, 'center_x': -1.5, 'center_y': -1.5})
kwargs_upper_lens.append({'theta_E': 5.0, 'e1': 1, 'e2': 1, 'center_x': 1.5, 'center_y': 1.5})

if 'SHEAR' in lens_model_list:
    fixed_lens.append({'ra_0': 0, 'dec_0': 0})
    kwargs_lens_init.append({'ra_0': 0, 'dec_0': 0,'gamma1': 0., 'gamma2': 0.0})
    kwargs_lens_sigma.append({'gamma1': 0.1, 'gamma2': 0.1})
    kwargs_lower_lens.append({'gamma1': -0.2, 'gamma2': -0.2})
    kwargs_upper_lens.append({'gamma1': 0.2, 'gamma2': 0.2})


# source
fixed_source = []
kwargs_source_init = []
kwargs_source_sigma = []
kwargs_lower_source = []
kwargs_upper_source = []

fixed_source.append({})
kwargs_source_init.append({'R_sersic': 0.2, 'n_sersic': 1, 'e1': 0., 'e2': 0.,'center_x': 0., 'center_y': 0})
kwargs_source_sigma.append({'R_sersic': 0.1,'n_sersic': 0.5,'e1': 0.5, 'e2': 0.5, 'center_x': 0.01, 'center_y': 0.01})
kwargs_lower_source.append({ 'R_sersic': 0.001, 'n_sersic': .1,'e1': -1, 'e2': -1, 'center_x': -1.5, 'center_y': -1.5})
kwargs_upper_source.append({ 'R_sersic': 10., 'n_sersic': 10.,'e1': 1, 'e2': 1, 'center_x': 1.5, 'center_y': 1.5})


if 'SHAPELETS' in source_model_list:
    fixed_source.append({'n_max': -2})
    kwargs_source_init.append({'n_max': -2,'center_x': 0.01, 'center_y': 0.01, 'beta': 0.2})
    kwargs_source_sigma.append({'center_x': 0.01, 'center_y': 0.01, 'beta': 0.05})
    kwargs_lower_source.append({'center_x': -1.5, 'center_y': -1.5, 'beta': 0.05})
    kwargs_upper_source.append({'center_x': 1.5, 'center_y': 1.5, 'beta': 0.05})



#lens light
fixed_lens_light = []
kwargs_lens_light_init = []
kwargs_lens_light_sigma = []
kwargs_lower_lens_light = []
kwargs_upper_lens_light = []

fixed_lens_light.append({})
kwargs_lens_light_init.append({'R_sersic': 0.5, 'n_sersic': 2,'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0})
kwargs_lens_light_sigma.append({'R_sersic': 0.1, 'n_sersic': 0.1,'e1': 0.5, 'e2': 0.5,  'center_x': 0.01, 'center_y': 0.01})
kwargs_lower_lens_light.append({'R_sersic': 0.001, 'n_sersic': 0.1,'e1': -1, 'e2': -1, 'center_x': -1.5, 'center_y': -1.5})
kwargs_upper_lens_light.append({'R_sersic': 10., 'n_sersic': 10.,'e1': 1, 'e2': 1, 'center_x': 1.5, 'center_y': 1.5})


lens_light_params = [[],[],[],[],[]]
for i in range(len(kwargs_data)):
    lens_light_params[0].extend(deepcopy(kwargs_lens_light_init))
    lens_light_params[1].extend(deepcopy(kwargs_lens_light_sigma))
    lens_light_params[2].extend(deepcopy(fixed_lens_light))
    lens_light_params[3].extend(deepcopy(kwargs_lower_lens_light))
    lens_light_params[4].extend(deepcopy(kwargs_upper_lens_light))

    
source_params = [[],[],[],[],[]]
for i in range(len(kwargs_data)):
    source_params[0].extend(deepcopy(kwargs_source_init))
    source_params[1].extend(deepcopy(kwargs_source_sigma))
    source_params[2].extend(deepcopy(fixed_source))
    source_params[3].extend(deepcopy(kwargs_lower_source))
    source_params[4].extend(deepcopy(kwargs_upper_source))

lens_params = [kwargs_lens_init, kwargs_lens_sigma, fixed_lens, kwargs_lower_lens, kwargs_upper_lens]

kwargs_params = {'lens_model': lens_params,
                'source_model': source_params,
                'lens_light_model': lens_light_params}

kwargs_fixed = {'kwargs_lens': fixed_lens, 
                 'kwargs_source': deepcopy(source_params[2]), 
                 'kwargs_lens_light': deepcopy(lens_light_params[2])}
