from Lens_Modeling_Auto.auto_modeling_functions import optParams
from Lens_Modeling_Auto.auto_modeling_functions import removekeys
from Lens_Modeling_Auto.auto_modeling_functions import prepareFit
from Lens_Modeling_Auto.auto_modeling_functions import runFit
from Lens_Modeling_Auto.auto_modeling_functions import get_kwarg_names
from lenstronomy.Workflow.fitting_sequence import FittingSequence


############ Set parameters to optimize (as a list if multiple PSOs are desired) ############

model_kwarg_names = get_kwarg_names(lens_model_list,source_model_list,lens_light_model_list,None)
opt_params = [{'kwargs_lens': [['center_x','center_y'],[]],
                       'kwargs_source': [[]],
                       'kwargs_lens_light': [[]]},
              {'kwargs_lens': [[],[]],
                       'kwargs_source': [['center_x','center_y']],
                       'kwargs_lens_light': [[]]},
              {'kwargs_lens': [[],[]],
                       'kwargs_source': [[]],
                       'kwargs_lens_light': [['center_x','center_y']]}]

for i in range(len(opt_params)):
    print('Free parameters:', opt_params[i])
    
    kwargs_init, kwargs_fixed = optParams(kwargs_result,opt_params[i],model_kwarg_names)

    lens_params = [kwargs_init['kwargs_lens'], kwargs_lens_sigma, kwargs_fixed['kwargs_lens'], kwargs_lower_lens,
                   kwargs_upper_lens]
    source_params = [kwargs_init['kwargs_source'], kwargs_source_sigma, kwargs_fixed['kwargs_source'], kwargs_lower_source,
                     kwargs_upper_source]
    lens_light_params = [kwargs_init['kwargs_lens_light'], kwargs_lens_light_sigma, kwargs_fixed['kwargs_lens_light'],
                         kwargs_lower_lens_light, kwargs_upper_lens_light]
    kwargs_params = {'lens_model': lens_params,
                    'source_model': source_params,
                    'lens_light_model': lens_light_params}

    print('The lens, source, and lens light modeling parameters are')
    print('lens model: ', kwargs_params['lens_model'])
    print('\n')
    print('source model: ', kwargs_params['source_model'])
    print('\n')
    print('lens light model: ', kwargs_params['lens_light_model'])
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    print('I will now begin the PSO:')

    fitting_kwargs_list = [['PSO', {'sigma_scale': 0.5, 'n_particles': 300, 'n_iterations': 1000,'threadCount': 1}]]

    kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list, kwargs_constraints = prepareFit(kwargs_data, kwargs_psf,
                                                                                 lens_model_list, source_model_list,
                                                                                 lens_light_model_list, 
                                                                                 image_mask_list = mask_list)                       
    chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                       kwargs_likelihood, kwargs_model,
                                       kwargs_data_joint, kwargs_constraints = kwargs_constraints) 