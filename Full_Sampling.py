from Lens_Modeling_Auto.auto_modeling_functions import optParams
from Lens_Modeling_Auto.auto_modeling_functions import removekeys
from Lens_Modeling_Auto.auto_modeling_functions import prepareFit
from Lens_Modeling_Auto.auto_modeling_functions import runFit
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from copy import deepcopy
                      

kwargs_params = {'lens_model': lens_params_update,
                'source_model': source_params_update,
                'lens_light_model': lens_light_params_update}
                       
                       
print('The lens, source, and lens light modeling parameters are')
print('lens model: ', kwargs_params['lens_model'])
print('\n')
print('source model: ', kwargs_params['source_model'])
print('\n')
print('lens light model: ', kwargs_params['lens_light_model'])
print('\n')
print('-------------------------------------------------------------------')
print('\n')
print('I will now begin the sampling')                      
                       
                       
#prepare fitting kwargs
kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list, kwargs_constraints = prepareFit(kwargs_data, kwargs_psf,
                                                                                 lens_model_list, source_model_list,
                                                                                 lens_light_model_list, 
                                                                                 image_mask_list = mask_list)                       
chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                   kwargs_likelihood, kwargs_model,
                                   kwargs_data_joint, kwargs_constraints = kwargs_constraints)         
                       
                       
                       
                    