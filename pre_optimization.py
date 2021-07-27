from Lens_Modeling_Auto.auto_modeling_functions import optParams
from Lens_Modeling_Auto.auto_modeling_functions import removekeys
from Lens_Modeling_Auto.auto_modeling_functions import prepareFit
from Lens_Modeling_Auto.auto_modeling_functions import runFit
from lenstronomy.Workflow.fitting_sequence import FittingSequence

model_kwarg_names = {'kwargs_lens': [['theta_E','e1','e2','center_x','center_y'],['gamma1','gamma2','ra_0','dec_0']],
                    'kwargs_source': [['amp','R_sersic','n_sersic','center_x','center_y']],
                    'kwargs_lens_light':[['amp','R_sersic','n_sersic','center_x','center_y']]}


#################Optimize Positions##############################
print('I will first optimize the lens centroid')
print('\n')
#Optimize lens Position
opt_params = {'kwargs_lens': [['center_x','center_y'],[]],
             'kwargs_source': [[]],
             'kwargs_lens_light': [[]]}

print('\n')
print('Free parameters:', opt_params)
      
kwargs_init, kwargs_fixed = optParams(kwargs_result,opt_params,model_kwarg_names)

lens_params = [kwargs_init['kwargs_lens'], kwargs_lens_sigma, kwargs_fixed['kwargs_lens'], kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_init['kwargs_source'], kwargs_source_sigma, kwargs_fixed['kwargs_source'], kwargs_lower_source, kwargs_upper_source]
lens_light_params = [kwargs_init['kwargs_lens_light'], kwargs_lens_light_sigma, kwargs_fixed['kwargs_lens_light'], kwargs_lower_lens_light, kwargs_upper_lens_light]
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

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 300, 'n_iterations': 1000,'threadCount': 1}]]

chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                   kwargs_likelihood, kwargs_model,
                                   kwargs_data_joint, kwargs_constraints = {})


print('\n')
print('##########################################################################')
print('\n')

print('I will now optimize the source centroid')
print('\n')
#Optimize Source Position
opt_params = {'kwargs_lens': [[],[]],
             'kwargs_source': [['center_x','center_y']],
             'kwargs_lens_light': [[]]}
print('\n')     
print('Free parameters:', opt_params)
      
kwargs_init, kwargs_fixed = optParams(kwargs_result,opt_params,model_kwarg_names)

lens_params = [kwargs_init['kwargs_lens'], kwargs_lens_sigma, kwargs_fixed['kwargs_lens'], kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_init['kwargs_source'], kwargs_source_sigma, kwargs_fixed['kwargs_source'], kwargs_lower_source, kwargs_upper_source]
lens_light_params = [kwargs_init['kwargs_lens_light'], kwargs_lens_light_sigma, kwargs_fixed['kwargs_lens_light'], kwargs_lower_lens_light, kwargs_upper_lens_light]
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

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 300, 'n_iterations': 1000,'threadCount': 1}]]

chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                   kwargs_likelihood, kwargs_model,
                                   kwargs_data_joint, kwargs_constraints = {})


print('\n')
print('##########################################################################')
print('\n')
      
print('I will now optimize the lens light centroid')
print('\n')
#Optimize lens light Position
opt_params = {'kwargs_lens': [[],[]],
             'kwargs_source': [[]],
             'kwargs_lens_light': [['center_x','center_y']]}
print('\n')      
print('Free parameters:', opt_params)
      
kwargs_init, kwargs_fixed = optParams(kwargs_result,opt_params,model_kwarg_names)

lens_params = [kwargs_init['kwargs_lens'], kwargs_lens_sigma, kwargs_fixed['kwargs_lens'], kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_init['kwargs_source'], kwargs_source_sigma, kwargs_fixed['kwargs_source'], kwargs_lower_source, kwargs_upper_source]
lens_light_params = [kwargs_init['kwargs_lens_light'], kwargs_lens_light_sigma, kwargs_fixed['kwargs_lens_light'], kwargs_lower_lens_light, kwargs_upper_lens_light]
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

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 300, 'n_iterations': 1000,'threadCount': 1}]]

chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                   kwargs_likelihood, kwargs_model,
                                   kwargs_data_joint, kwargs_constraints = {})

print('These positions will remain fixed while I now optimize the other parameters') 


#####################################Optimize Params###########################################


print('\n')
print('##########################################################################')
print('\n')

     
      
print('I will now optimize the lens parameters')
print('\n')
#Optimize lens params
opt_params = {'kwargs_lens': [['theta_E','e1','e2'],[]],
             'kwargs_source': [[]],
             'kwargs_lens_light': [[]]}
print('\n')      
print('Free parameters:', opt_params)
      
kwargs_init, kwargs_fixed = optParams(kwargs_result,opt_params,model_kwarg_names)

lens_params = [kwargs_init['kwargs_lens'], kwargs_lens_sigma, kwargs_fixed['kwargs_lens'], kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_init['kwargs_source'], kwargs_source_sigma, kwargs_fixed['kwargs_source'], kwargs_lower_source, kwargs_upper_source]
lens_light_params = [kwargs_init['kwargs_lens_light'], kwargs_lens_light_sigma, kwargs_fixed['kwargs_lens_light'], kwargs_lower_lens_light, kwargs_upper_lens_light]
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

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 300, 'n_iterations': 1000,'threadCount': 1}]]

chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                   kwargs_likelihood, kwargs_model,
                                   kwargs_data_joint, kwargs_constraints = {})


print('\n')
print('##########################################################################')
print('\n')
      
      
print('I will now optimize the source parameters')
print('\n')
#Optimize source params
opt_params = {'kwargs_lens': [[],[]],
             'kwargs_source': [['amp','R_sersic','n_sersic']],
             'kwargs_lens_light': [[]]}
print('\n')   
print('Free parameters:', opt_params)
      
kwargs_init, kwargs_fixed = optParams(kwargs_result,opt_params,model_kwarg_names)

lens_params = [kwargs_init['kwargs_lens'], kwargs_lens_sigma, kwargs_fixed['kwargs_lens'], kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_init['kwargs_source'], kwargs_source_sigma, kwargs_fixed['kwargs_source'], kwargs_lower_source, kwargs_upper_source]
lens_light_params = [kwargs_init['kwargs_lens_light'], kwargs_lens_light_sigma, kwargs_fixed['kwargs_lens_light'], kwargs_lower_lens_light, kwargs_upper_lens_light]
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

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 300, 'n_iterations': 1000,'threadCount': 1}]]

chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                   kwargs_likelihood, kwargs_model,
                                   kwargs_data_joint, kwargs_constraints = {})


print('\n')
print('##########################################################################')
print('\n')

      
print('I will now optimize the lens light parameters')
print('\n')
#Optimize lens light params
opt_params = {'kwargs_lens': [[],[]],
             'kwargs_source': [[]],
             'kwargs_lens_light': [['amp','R_sersic','n_sersic']]}
print('\n')      
print('Free parameters:', opt_params)
      
kwargs_init, kwargs_fixed = optParams(kwargs_result,opt_params,model_kwarg_names)

lens_params = [kwargs_init['kwargs_lens'], kwargs_lens_sigma, kwargs_fixed['kwargs_lens'], kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_init['kwargs_source'], kwargs_source_sigma, kwargs_fixed['kwargs_source'], kwargs_lower_source, kwargs_upper_source]
lens_light_params = [kwargs_init['kwargs_lens_light'], kwargs_lens_light_sigma, kwargs_fixed['kwargs_lens_light'], kwargs_lower_lens_light, kwargs_upper_lens_light]
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

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 300, 'n_iterations': 1000,'threadCount': 1}]]

chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                   kwargs_likelihood, kwargs_model,
                                   kwargs_data_joint, kwargs_constraints = {})


print('\n')
print('##########################################################################')
print('\n')
      
      
print('I will now optimize the lens + lens light parameters')
print('\n')
#Optimize lens + light params
opt_params = {'kwargs_lens': [['theta_E','e1','e2'],[]],
             'kwargs_source': [[]],
             'kwargs_lens_light': [['amp','R_sersic','n_sersic']]}
print('\n')      
print('Free parameters:', opt_params)
      
kwargs_init, kwargs_fixed = optParams(kwargs_result,opt_params,model_kwarg_names)

lens_params = [kwargs_init['kwargs_lens'], kwargs_lens_sigma, kwargs_fixed['kwargs_lens'], kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_init['kwargs_source'], kwargs_source_sigma, kwargs_fixed['kwargs_source'], kwargs_lower_source, kwargs_upper_source]
lens_light_params = [kwargs_init['kwargs_lens_light'], kwargs_lens_light_sigma, kwargs_fixed['kwargs_lens_light'], kwargs_lower_lens_light, kwargs_upper_lens_light]
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

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 300, 'n_iterations': 1000,'threadCount': 1}]]

chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                   kwargs_likelihood, kwargs_model,
                                   kwargs_data_joint, kwargs_constraints = {})


print('\n')
print('##########################################################################')
print('\n')
      

print('I will now optimize the lens + lens light + Source parameters')
print('\n')
#Optimize lens + light params
opt_params = {'kwargs_lens': [['theta_E','e1','e2'],[]],
             'kwargs_source': [['amp','R_sersic','n_sersic']],
             'kwargs_lens_light': [['amp','R_sersic','n_sersic']]}
print('\n')      
print('Free parameters:', opt_params)
      
kwargs_init, kwargs_fixed = optParams(kwargs_result,opt_params,model_kwarg_names)

lens_params = [kwargs_init['kwargs_lens'], kwargs_lens_sigma, kwargs_fixed['kwargs_lens'], kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_init['kwargs_source'], kwargs_source_sigma, kwargs_fixed['kwargs_source'], kwargs_lower_source, kwargs_upper_source]
lens_light_params = [kwargs_init['kwargs_lens_light'], kwargs_lens_light_sigma, kwargs_fixed['kwargs_lens_light'], kwargs_lower_lens_light, kwargs_upper_lens_light]
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

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 300, 'n_iterations': 1000,'threadCount': 1}]]

chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                   kwargs_likelihood, kwargs_model,
                                   kwargs_data_joint, kwargs_constraints = {})


print('\n')
print('##########################################################################')
print('\n')