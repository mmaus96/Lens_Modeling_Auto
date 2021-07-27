import numpy as np
import os, psutil
from os import walk
from os import listdir
from os.path import isfile, join, isdir, exists
from Lens_Modeling_Auto.auto_modeling_functions import prepareFit
from Lens_Modeling_Auto.auto_modeling_functions import runFit
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from Lens_Modeling_Auto.auto_modeling_functions import find_components
from Lens_Modeling_Auto.auto_modeling_functions import mask_for_sat
from Lens_Modeling_Auto.auto_modeling_functions import mask_for_lens_gal
from Lens_Modeling_Auto.plot_functions import plot_LRG_fit
from Lens_Modeling_Auto.plot_functions import plot_lensed_source_fit

import pickle
from copy import deepcopy

#################################################################################################

######################################### Initial Params ########################################

#################################################################################################

def initial_model_params(lens_model_list,source_model_list = ['SERSIC_ELLIPSE'],
                         lens_light_model_list = ['SERSIC_ELLIPSE'], point_source_model_list = None,n_max=-1):
    lens_initialization = {}

    lens_initialization['SIE'] = {}
    lens_initialization['SIE']['fixed'] = {}
    lens_initialization['SIE']['init'] = {'theta_E': 1.5, 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.}
    lens_initialization['SIE']['sigma'] = {'theta_E': .3, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.1, 'center_y': 0.1}
    lens_initialization['SIE']['lower'] = {'theta_E': 0.1, 'e1': -1, 'e2': -1, 'center_x': -1.5, 'center_y': -1.5}
    lens_initialization['SIE']['upper'] = {'theta_E': 5.0, 'e1': 1, 'e2': 1, 'center_x': 1.5, 'center_y': 1.5}

    lens_initialization['PEMD'] = {}
    lens_initialization['PEMD']['fixed'] = {}
    lens_initialization['PEMD']['init'] = {'theta_E': 1.5, 'gamma': 2., 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.}
    lens_initialization['PEMD']['sigma'] = {'theta_E': .3, 'gamma': 0.1,'e1': 0.5, 'e2': 0.5, 'center_x': 0.1, 'center_y': 0.1}
    lens_initialization['PEMD']['lower'] = {'theta_E': 0.1, 'gamma': 1.4,'e1': -1, 'e2': -1, 'center_x': -1.5, 'center_y': -1.5}
    lens_initialization['PEMD']['upper'] = {'theta_E': 5.0, 'gamma': 2.8, 'e1': 1, 'e2': 1, 'center_x': 1.5, 'center_y': 1.5}

    lens_initialization['SHEAR'] = {}
    lens_initialization['SHEAR']['fixed'] = {'ra_0': 0, 'dec_0': 0}
    lens_initialization['SHEAR']['init'] = {'ra_0': 0, 'dec_0': 0,'gamma1': 0., 'gamma2': 0.0}
    lens_initialization['SHEAR']['sigma'] = {'gamma1': 0.1, 'gamma2': 0.1}
    lens_initialization['SHEAR']['lower'] = {'gamma1': -0.5, 'gamma2': -0.5}
    lens_initialization['SHEAR']['upper'] = {'gamma1': 0.5, 'gamma2': 0.5}

    source_initialization = {}

    source_initialization['SERSIC_ELLIPSE'] = {}
    source_initialization['SERSIC_ELLIPSE']['fixed'] = {}
#     source_initialization['SERSIC']['init'] = {'amp': 1.,'R_sersic': 1.0, 'n_sersic': 1.,'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.}
    source_initialization['SERSIC_ELLIPSE']['init'] = {'R_sersic': 0.4, 'n_sersic': 1.0,'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0}
    source_initialization['SERSIC_ELLIPSE']['sigma'] = {'R_sersic': 0.5, 'n_sersic': 0.5,'e1': 0.5, 'e2': 0.5,  'center_x': 0.1, 'center_y': 0.1}
    
#     source_initialization['SERSIC']['sigma'] = {'R_sersic': 0.1, 'n_sersic': 0.5,'e1': 0.5, 'e2': 0.5,  'center_x': 0.01, 'center_y': 0.01}
    
    source_initialization['SERSIC_ELLIPSE']['lower'] = {'R_sersic': 0.001, 'n_sersic': 0.1,'e1': -0.5, 'e2': -0.5, 'center_x': -1.5, 'center_y': -1.5}
    source_initialization['SERSIC_ELLIPSE']['upper'] = {'R_sersic':1.2, 'n_sersic': 6.,'e1': 0.5, 'e2': 0.5, 'center_x': 1.5, 'center_y': 1.5}
    
    source_initialization['SHAPELETS'] = {}
    source_initialization['SHAPELETS']['fixed'] = {'n_max': n_max}
    source_initialization['SHAPELETS']['init'] = {'center_x': 0.0, 'center_y': 0.0, 'beta': 0.1}
    source_initialization['SHAPELETS']['sigma'] = {'center_x': 0.01, 'center_y': 0.01, 'beta': 0.01}
    source_initialization['SHAPELETS']['lower'] = {'center_x': -1.5, 'center_y': -1.5, 'beta': 0.1 / np.sqrt(n_max + 1)}
    source_initialization['SHAPELETS']['upper'] = {'center_x': 1.5, 'center_y': 1.5, 'beta': 0.1 * np.sqrt(n_max + 1)}

    lens_light_initialization = {}

    lens_light_initialization['SERSIC_ELLIPSE'] = {}
    lens_light_initialization['SERSIC_ELLIPSE']['fixed'] = {}
#     lens_light_initialization['SERSIC']['init'] = {'amp': 1., 'R_sersic': 1.0, 'n_sersic': 2.,'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.}
    lens_light_initialization['SERSIC_ELLIPSE']['init'] = {'R_sersic': 1.0, 'n_sersic': 4.0,'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0}
#     lens_light_initialization['SERSIC']['sigma'] = {'R_sersic': 0.1, 'n_sersic': 0.5,'e1': 0.5, 'e2': 0.5,  'center_x': 0.01, 'center_y': 0.01}
    lens_light_initialization['SERSIC_ELLIPSE']['sigma'] = {'R_sersic': 0.5, 'n_sersic': 0.5,'e1': 0.5, 'e2': 0.5,  'center_x': 0.1, 'center_y': 0.1}
    lens_light_initialization['SERSIC_ELLIPSE']['lower'] = {'R_sersic': 0.001, 'n_sersic': 0.1,'e1': -0.5, 'e2': -0.5, 'center_x': -1.5, 'center_y': -1.5}
    lens_light_initialization['SERSIC_ELLIPSE']['upper'] = {'R_sersic': 5., 'n_sersic': 10.,'e1': 0.5, 'e2': 0.5, 'center_x': 1.5, 'center_y': 1.5}


    lens_initial_params = [[],[],[],[],[]]
    for prof in lens_model_list:
        lens_initial_params[0].append(deepcopy(lens_initialization[prof]['init']))
        lens_initial_params[1].append(deepcopy(lens_initialization[prof]['sigma']))
        lens_initial_params[2].append(deepcopy(lens_initialization[prof]['fixed']))
        lens_initial_params[3].append(deepcopy(lens_initialization[prof]['lower']))
        lens_initial_params[4].append(deepcopy(lens_initialization[prof]['upper']))


    lens_light_initial_params = deepcopy([lens_light_initialization['SERSIC_ELLIPSE']['init'], 
                                          lens_light_initialization['SERSIC_ELLIPSE']['sigma'], 
                                          lens_light_initialization['SERSIC_ELLIPSE']['fixed'], 
                                          lens_light_initialization['SERSIC_ELLIPSE']['lower'], 
                                          lens_light_initialization['SERSIC_ELLIPSE']['upper']])

    if len(source_model_list) >1:
        source_initial_params = [[],[],[],[],[]]
        for prof in source_model_list:
            source_initial_params[0].append(deepcopy(source_initialization[prof]['init']))
            source_initial_params[1].append(deepcopy(source_initialization[prof]['sigma']))
            source_initial_params[2].append(deepcopy(source_initialization[prof]['fixed']))
            source_initial_params[3].append(deepcopy(source_initialization[prof]['lower']))
            source_initial_params[4].append(deepcopy(source_initialization[prof]['upper']))
    else:
        source_initial_params = deepcopy([source_initialization['SERSIC_ELLIPSE']['init'], 
                                          source_initialization['SERSIC_ELLIPSE']['sigma'], 
                                          source_initialization['SERSIC_ELLIPSE']['fixed'], 
                                          source_initialization['SERSIC_ELLIPSE']['lower'], 
                                          source_initialization['SERSIC_ELLIPSE']['upper']])
    
    
    if point_source_model_list != None:
        ps_init = {}
        ps_init['init'] = {'ra_source':0,'dec_source':0}
        ps_init['sigma'] = {'ra_source':0.1,'dec_source':0.1}
        ps_init['fixed'] = {}
        ps_init['lower'] = {'ra_source':-1.5,'dec_source':-1.5}
        ps_init['upper'] = {'ra_source':1.5,'dec_source':1.5}
        
        ps_initial_params = [ps_init['init'],ps_init['sigma'],ps_init['fixed'],ps_init['lower'],ps_init['upper']]
    else: 
        ps_initial_params = None
        

    
    return lens_initial_params,source_initial_params,lens_light_initial_params,ps_initial_params
    

#################################################################################################

##################################### Initial Modeling Fits ######################################

#################################################################################################

def initial_modeling_fit(fitting_kwargs_list,lens_model_list,source_model_list,lens_light_model_list,
                        lens_initial_params,source_initial_params,lens_light_initial_params,
                        kwargs_data,kwargs_psf,mask_list,gal_mask_list,fix_seed = False, fix_seed_val = None, 
                        save_seed_file = None, chainList_file = None):
    
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for l,x in enumerate(source_params):
        for i in range(len(kwargs_data)):
            x.append(deepcopy(source_initial_params[l]))


    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data)):
            f.append(deepcopy(lens_light_initial_params[j]))


    lens_params = deepcopy(lens_initial_params)
    
    ########################################## Optimize All ##########################################
    print('I will now optimize Everything')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.

     
    if fix_seed:
        np.random.set_state(fix_seed_val)
    else:
        np.random.seed(None)
        
    
    # get the initial state of the RNG
    save_seed_val = np.random.get_state()

    with open(save_seed_file, 'wb') as handle:
            pickle.dump(save_seed_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params,lens_light_params, 
                                                                      image_mask_list = mask_list)
    
    
    
    with open(chainList_file, 'wb') as handle:
            pickle.dump(chain_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('\n')
    print('##########################################################################')
    print('\n')


    ########################################## Lens Light ##########################################


    print('I will now fit the lens galaxy by masking lensed arcs and fixing source and lens parameters')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
#     fitting_kwargs_list[0][1]['sigma_scale'] = 0.8
#     print('dropping sigma_scale to 0.8')
    print('\n')
    
    lens_params[2] = deepcopy(lens_params[0])
    source_params[2] = deepcopy(source_params[0])
    
    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
        
        
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      image_mask_list = gal_mask_list)

          
    ########################################## Optimize Lens + Source ##########################################
    print('\n')
    print('##########################################################################')
    print('\n')
    print('I will now optimize the lens profile and source light together')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    #Fix Lens Light
    lens_light_params[2] = deepcopy(kwargs_result['kwargs_lens_light']) 
    source_params[2] = []
    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      image_mask_list = mask_list)


    ########################################## Optimize All ##########################################
    print('I will now optimize Everything')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    lens_light_params[2] = []
    source_params[2] = []
    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
        lens_light_params[2].append(deepcopy(lens_light_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      image_mask_list = mask_list) 

    
   ########################################## Lens Light (2) ##########################################


    print('I will again fit the lens galaxy by masking lensed arcs and fixing source and lens parameters (round 2)')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
#     fitting_kwargs_list[0][1]['sigma_scale'] = 0.5
#     print('dropping sigma_scale to 0.5')
    print('\n')
    
    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
        
    lens_params[2] = deepcopy(kwargs_result['kwargs_lens']) 
    source_params[2] = deepcopy(kwargs_result['kwargs_source']) 
        
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params,source_params, lens_light_params, 
                                                                      image_mask_list = gal_mask_list) 
    
    
    
    ########################################## Optimize Lens + Source (2) ##########################################
    print('\n')
    print('##########################################################################')
    print('\n')
    print('I will again optimize the lens profile and source light together (round 2)')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    #Fix Lens Light
    lens_light_params[2] = deepcopy(kwargs_result['kwargs_lens_light']) 
    source_params[2] = []
    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      image_mask_list = mask_list)


    ########################################## Optimize All (2) ##########################################
    print('I will now optimize Everything (round 2)')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    lens_light_params[2] = []
    source_params[2] = []
    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
        lens_light_params[2].append(deepcopy(lens_light_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      image_mask_list = mask_list) 
    
    
    lens_params[2] = deepcopy(lens_initial_params[2])
    
    kwargs_params = {'lens_model': deepcopy(lens_params),
                    'source_model': deepcopy(source_params),
                    'lens_light_model': deepcopy(lens_light_params)}
    
    kwargs_fixed = {'kwargs_lens': deepcopy(lens_params[2]), 
                     'kwargs_source': deepcopy(source_params[2]), 
                     'kwargs_lens_light': deepcopy(lens_light_params[2])}
    
    return (kwargs_params,kwargs_fixed, kwargs_result,
            chain_list,kwargs_likelihood, kwargs_model, 
            kwargs_data_joint, multi_band_list, kwargs_constraints)



#################################################################################################

##################################### Initial Modeling Fits ######################################

#################################################################################################

def run_fit(fitting_kwargs_list,kwargs_data, kwargs_psf,lens_model_list, source_model_list,lens_light_model_list,
            lens_params, source_params, lens_light_params,ps_model_list = None,
            ps_params=None,kde_nsource=None,kde_Rsource=None,image_mask_list = None):
    
    #prepare fitting kwargs
    kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list,kwargs_constraints = prepareFit(kwargs_data, kwargs_psf,
                                                                                     lens_model_list, source_model_list,
                                                                                     lens_light_model_list,
                                                                                     kde_nsource = kde_nsource,
                                                                                     kde_Rsource = kde_Rsource,
                                                                                     ps_model_list = ps_model_list,
                                                                                     check_pos_flux = True,
                                                                                     image_mask_list = image_mask_list)
    kwargs_params = {'lens_model': deepcopy(lens_params),
                    'source_model': deepcopy(source_params),
                    'lens_light_model': deepcopy(lens_light_params)}
    
    if ps_model_list != None:
        kwargs_params['point_source_model'] = deepcopy(ps_params)

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


    chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                       kwargs_likelihood, kwargs_model,
                                       kwargs_data_joint, kwargs_constraints = kwargs_constraints)
    
    print('\n')
    print('##########################################################################')
    print('\n')



    lens_light_params[0] = deepcopy(kwargs_result['kwargs_lens_light'])                                

    source_params[0] = deepcopy(kwargs_result['kwargs_source'])                               

    lens_params[0] = deepcopy(kwargs_result['kwargs_lens'])
    
    

    kwargs_params = {'lens_model': deepcopy(lens_params),
                    'source_model': deepcopy(source_params),
                    'lens_light_model': deepcopy(lens_light_params)}
    
    if ps_model_list != None:
        ps_params[0] = deepcopy(kwargs_result['kwargs_ps'])
        kwargs_params['point_source_model'] = deepcopy(ps_params)

  
    return (kwargs_params, lens_params,source_params,lens_light_params, ps_params,
            kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
            kwargs_data_joint, multi_band_list,kwargs_constraints)


    print('\n')
    print('##########################################################################')
    print('\n')
    

##############################################################################################################

##################################### Initial Modeling Fits-Masked arcs ######################################

##############################################################################################################    
    
def initial_fits_arcs_masked(fitting_kwargs_list,lens_model_list,source_model_list,lens_light_model_list,
                        lens_initial_params,source_initial_params,lens_light_initial_params,
                        kwargs_data,kwargs_psf,mask_list,gal_mask_list, ps_model_list = None, ps_initial_params=None,
                        kde_nsource=None,kde_Rsource=None,
                        fix_seed = False, fix_seed_val = None, 
                        save_seed_file = None, chainList_file = None):
    
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]
    ps_params = [[],[],[],[],[]]
                            

    for l,x in enumerate(source_params):
        for i in range(len(kwargs_data)):
            x.append(deepcopy(source_initial_params[l]))


    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data)):
            f.append(deepcopy(lens_light_initial_params[j]))
    
    if ps_model_list !=None:
        for k,y in enumerate(ps_params):
            for i in range(len(kwargs_data)):
                y.append(deepcopy(ps_initial_params[k]))


    lens_params = deepcopy(lens_initial_params)
    
    ########################################## Optimize All ##########################################
    print('I will now optimize Everything')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.

     
    if fix_seed:
        np.random.set_state(fix_seed_val)
    else:
        np.random.seed(None)
        
    
    # get the initial state of the RNG
    save_seed_val = np.random.get_state()

    with open(save_seed_file, 'wb') as handle:
            pickle.dump(save_seed_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    (kwargs_params, lens_params,source_params,lens_light_params, ps_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params,lens_light_params, 
                                                                      ps_model_list = ps_model_list, ps_params = ps_params,
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, 
                                                                      image_mask_list = mask_list)
    
    
    
    with open(chainList_file, 'wb') as handle:
            pickle.dump(chain_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ########################################## Lens Light ##########################################


    print('I will now fit the lens galaxy by masking lensed arcs and fixing source and lens parameters')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
#     fitting_kwargs_list[0][1]['sigma_scale'] = 0.8
#     print('dropping sigma_scale to 0.8')
    print('\n')
    
    lens_params[2] = deepcopy(lens_params[0])
    source_params[2] = deepcopy(source_params[0])
    if ps_model_list !=None:
        ps_params[2] = deepcopy(ps_params[0])
    
    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
        
        
    (kwargs_params, lens_params,source_params,lens_light_params, ps_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      ps_model_list = ps_model_list, ps_params = ps_params,
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, 
                                                                      image_mask_list = gal_mask_list)

          
    ########################################## Optimize Lens + Source ##########################################
    print('\n')
    print('##########################################################################')
    print('\n')
    print('I will now optimize the lens profile and source light together')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    #Fix Lens Light
    lens_light_params[2] = deepcopy(kwargs_result['kwargs_lens_light']) 
    source_params[2] = []
    ps_params[2] = []
    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
        if ps_model_list !=None:
            ps_params[2].append(deepcopy(ps_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params, ps_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params,lens_light_params, 
                                                                      ps_model_list = ps_model_list, ps_params = ps_params,
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, 
                                                                      image_mask_list = mask_list)


    ########################################## Optimize All ##########################################
    print('I will now optimize Everything')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    lens_light_params[2] = []
    source_params[2] = []
    ps_params[2] = []

    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
        lens_light_params[2].append(deepcopy(lens_light_initial_params[2]))
        if ps_model_list !=None:
            ps_params[2].append(deepcopy(ps_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params, ps_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params,lens_light_params, 
                                                                      ps_model_list = ps_model_list, ps_params = ps_params,
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, 
                                                                      image_mask_list = mask_list)

    
   ########################################## Lens Light (2) ##########################################


    print('I will again fit the lens galaxy by masking lensed arcs and fixing source and lens parameters (round 2)')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
#     fitting_kwargs_list[0][1]['sigma_scale'] = 0.5
#     print('dropping sigma_scale to 0.5')
    print('\n')
    
    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
        
    lens_params[2] = deepcopy(kwargs_result['kwargs_lens']) 
    source_params[2] = deepcopy(kwargs_result['kwargs_source'])
    if ps_model_list !=None:
        ps_params[2] = deepcopy(kwargs_result['kwargs_ps']) 
        
    (kwargs_params, lens_params,source_params,lens_light_params, ps_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      ps_model_list = ps_model_list, ps_params = ps_params,
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, 
                                                                      image_mask_list = gal_mask_list) 
    
    
    
    ########################################## Optimize Lens + Source (2) ##########################################
    print('\n')
    print('##########################################################################')
    print('\n')
    print('I will again optimize the lens profile and source light together (round 2)')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    #Fix Lens Light
    lens_light_params[2] = deepcopy(kwargs_result['kwargs_lens_light']) 
    source_params[2] = []
    ps_params[2] = []
    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
        if ps_model_list !=None:
            ps_params[2].append(deepcopy(ps_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params, ps_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      ps_model_list = ps_model_list, ps_params = ps_params,
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource,  
                                                                      image_mask_list = mask_list)


    ########################################## Optimize All (2) ##########################################
    print('I will now optimize Everything (round 2)')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    lens_light_params[2] = []
    source_params[2] = []
    ps_params[2] = []

    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
        lens_light_params[2].append(deepcopy(lens_light_initial_params[2]))
        if ps_model_list !=None:
            ps_params[2].append(deepcopy(ps_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params, ps_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      ps_model_list = ps_model_list, ps_params = ps_params,
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource,  
                                                                      image_mask_list = mask_list) 
    
    
    lens_params[2] = deepcopy(lens_initial_params[2])
    
    kwargs_params = {'lens_model': deepcopy(lens_params),
                    'source_model': deepcopy(source_params),
                    'lens_light_model': deepcopy(lens_light_params)}
    
    if ps_model_list != None:
        kwargs_params['point_source_model'] = deepcopy(ps_params)
    
    kwargs_fixed = {'kwargs_lens': deepcopy(lens_params[2]), 
                     'kwargs_source': deepcopy(source_params[2]), 
                     'kwargs_lens_light': deepcopy(lens_light_params[2])}
    
    return (kwargs_params,kwargs_fixed, kwargs_result,
            chain_list,kwargs_likelihood, kwargs_model, 
            kwargs_data_joint, multi_band_list, kwargs_constraints)

##########################################################################################################################

##################################### Initial Modeling Fits-Masked arcs - Alternate ######################################

##########################################################################################################################

def initial_fits_arcs_masked_alt(fitting_kwargs_list,lens_model_list,source_model_list,lens_light_model_list,
                        lens_initial_params,source_initial_params,lens_light_initial_params,
                        kwargs_data,kwargs_psf,mask_list,gal_mask_list,
                        kde_nsource=None,kde_Rsource=None,
                        fix_seed = False, fix_seed_val = None, 
                        save_seed_file = None, chainList_file = None):
    
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]


    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data)):
            f.append(deepcopy(lens_light_initial_params[j]))

 
    
    ########################################## Lens Light ##########################################


    print('I will first fit the lens galaxy by masking lensed arcs and only define lens light sersic profile')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    
#     lens_params[2] = deepcopy(lens_params[0])
#     source_params[2] = deepcopy(source_params[0])
    
#     if 'PEMD' in lens_model_list:
#         lens_params[2][0]['gamma'] = 2.
    lens_model_list_init = []
    source_model_list_init = []
        
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list_init,
                                                                      source_model_list_init,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, 
                                                                      image_mask_list = gal_mask_list)

    
    
    ########################################## Optimize Lens + Source ##########################################
    
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for l,x in enumerate(source_params):
        for i in range(len(kwargs_data)):
            x.append(deepcopy(source_initial_params[l]))
    lens_params = deepcopy(lens_initial_params)
    
    
    print('\n')
    print('##########################################################################')
    print('\n')
    print('I will now optimize the lens profile and source light together')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    #Fix Lens Light
    lens_light_params[2] = deepcopy(kwargs_result['kwargs_lens_light']) 
    source_params[2] = []
    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, 
                                                                      image_mask_list = mask_list)


    ########################################## Optimize All ##########################################
    print('I will now optimize Everything')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    lens_light_params[2] = []
    source_params[2] = []
    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
        lens_light_params[2].append(deepcopy(lens_light_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, 
                                                                      image_mask_list = mask_list) 

    
   ########################################## Lens Light (2) ##########################################


    print('I will again fit the lens galaxy by masking lensed arcs and fixing source and lens parameters (round 2)')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
        
    lens_params[2] = deepcopy(kwargs_result['kwargs_lens']) 
    source_params[2] = deepcopy(kwargs_result['kwargs_source']) 
        
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params, source_params, lens_light_params, 
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, 
                                                                      image_mask_list = gal_mask_list) 
    
    
    
    ########################################## Optimize Lens + Source (2) ##########################################
    print('\n')
    print('##########################################################################')
    print('\n')
    print('I will again optimize the lens profile and source light together (round 2)')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    #Fix Lens Light
    lens_light_params[2] = deepcopy(kwargs_result['kwargs_lens_light']) 
    source_params[2] = []
    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params,source_params, lens_light_params, 
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, 
                                                                      image_mask_list = mask_list)


    ########################################## Optimize All (2) ##########################################
    print('I will now optimize Everything (round 2)')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')

    lens_light_params[2] = []
    source_params[2] = []
    for i in range(len(kwargs_data)):
        source_params[2].append(deepcopy(source_initial_params[2]))
        lens_light_params[2].append(deepcopy(lens_light_initial_params[2]))
    lens_params[2] = deepcopy(lens_initial_params[2])

    if 'PEMD' in lens_model_list:
        lens_params[2][0]['gamma'] = 2.
          
    (kwargs_params, lens_params,source_params,lens_light_params,
     kwargs_result,chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data, 
                                                                      kwargs_psf,lens_model_list,
                                                                      source_model_list,lens_light_model_list,
                                                                      lens_params,source_params, lens_light_params, 
                                                                      kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, 
                                                                      image_mask_list = mask_list) 
    
    
    lens_params[2] = deepcopy(lens_initial_params[2])
    
    kwargs_params = {'lens_model': deepcopy(lens_params),
                    'source_model': deepcopy(source_params),
                    'lens_light_model': deepcopy(lens_light_params)}
    
    kwargs_fixed = {'kwargs_lens': deepcopy(lens_params[2]), 
                     'kwargs_source': deepcopy(source_params[2]), 
                     'kwargs_lens_light': deepcopy(lens_light_params[2])}
    
    return (kwargs_params,kwargs_fixed, kwargs_result,
            chain_list,kwargs_likelihood, kwargs_model, 
            kwargs_data_joint, multi_band_list, kwargs_constraints)


##########################################################################################################################

############################################## Model Deblended Images ###################################################

##########################################################################################################################


def model_deblended(fitting_kwargs_list,lens_model_list,source_model_list,lens_light_model_list,
                        lens_initial_params,source_initial_params,lens_light_initial_params,
                        kwargs_data_lens,kwargs_data_LRG,kwargs_data_source,kwargs_psf,
                        num,object_ID,mask_list = None,kde_nsource=None,kde_Rsource=None,
                        source_mask_list = None, gal_mask_list =None,
                        fix_seed = False, fix_seed_val = None, 
                        save_seed_file = None, chainList_file = None,
                        results_path = None, band_list = None,):
    
    
    if fix_seed:
        np.random.set_state(fix_seed_val)
    else:
        np.random.seed(None)
        
    
    # get the initial state of the RNG
    save_seed_val = np.random.get_state()

    with open(save_seed_file, 'wb') as handle:
            pickle.dump(save_seed_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    print('I will first fit the LRG')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    # fit LRG:
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data_LRG)):
            f.append(deepcopy(lens_light_initial_params[j]))


    (kwargs_params, lens_params,source_params,
    lens_light_params,ps_params,kwargs_result_LRG,chain_list,
    kwargs_likelihood, kwargs_model, kwargs_data_joint, 
    multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_LRG, kwargs_psf,
                                                  lens_model_list = [],
                                                  source_model_list = [],
                                                  lens_light_model_list = lens_light_model_list,
                                                  lens_params = lens_params, 
                                                  source_params = source_params, 
                                                  lens_light_params = lens_light_params, 
                                                  kde_nsource=kde_nsource,kde_Rsource=kde_Rsource,
                                                  image_mask_list = mask_list)
    
    
    
    
    LRG_path = results_path + '/LRG/'
    LRG_plot_path = LRG_path + 'modelPlots/'
    if not exists(LRG_path):
        os.mkdir(LRG_path)
    if not exists(LRG_plot_path):
        os.mkdir(LRG_plot_path)
    
    
    LRG_plot_kwargs = {'multi_band_list': multi_band_list, 'kwargs_model': kwargs_model,
                   'kwargs_params': kwargs_result_LRG, 'likelihood_mask_list': mask_list}
    
    
    LRG_plot_kwargs = plot_LRG_fit(LRG_plot_kwargs,band_list,LRG_plot_path,num, object_ID)
    
    with open(LRG_plot_path + 'Image_{}-{}_plot_kwargs.pickle'.format(num,object_ID),'wb') as handle:
        pickle.dump(LRG_plot_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    print('I will now fit the lensed source')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    # fit lensed source
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for j,f in enumerate(source_params):
        for i in range(len(kwargs_data_source)):
            f.append(deepcopy(source_initial_params[j]))

    lens_params = deepcopy(lens_initial_params)
    
    
        

    (kwargs_params, lens_params,source_params,
    lens_light_params,ps_params,kwargs_result_source,chain_list,
    kwargs_likelihood, kwargs_model, kwargs_data_joint, 
    multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_source, kwargs_psf,
                                                  lens_model_list = lens_model_list,
                                                  source_model_list = source_model_list,
                                                  lens_light_model_list = [],
                                                  lens_params = lens_params, 
                                                  source_params = source_params, 
                                                  lens_light_params = lens_light_params, 
                                                  kde_nsource=kde_nsource,kde_Rsource=kde_Rsource,
                                                  image_mask_list = source_mask_list)
    
    source_path = results_path + '/lensed_source/'
    source_plot_path = source_path + 'modelPlots/'
    if not exists(source_path):
        os.mkdir(source_path)
    if not exists(source_plot_path):
        os.mkdir(source_plot_path)
    
    
    source_plot_kwargs = {'multi_band_list': multi_band_list, 'kwargs_model': kwargs_model,
                   'kwargs_params': kwargs_result_source, 'likelihood_mask_list': source_mask_list}
    
    
    source_plot_kwargs = plot_lensed_source_fit(source_plot_kwargs,kwargs_data_source, kwargs_psf,
                           band_list,lens_model_list,source_model_list,lens_light_model_list,
                           source_plot_path,num, object_ID)
    
    with open(source_plot_path + 'Image_{}-{}_plot_kwargs.pickle'.format(num,object_ID),'wb') as handle:
        pickle.dump(source_plot_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    kwargs_result = deepcopy(kwargs_result_source)
    kwargs_result['kwargs_lens_light'] = deepcopy(kwargs_result_LRG['kwargs_lens_light'])
    
    ########################################## Optimize All ##########################################
    print('I will now fit the original image')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for l,x in enumerate(source_params):
        for i in range(len(kwargs_data_lens)):
            x.append(deepcopy(source_initial_params[l]))


    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data_lens)):
            f.append(deepcopy(lens_light_initial_params[j]))
            
    lens_params = deepcopy(lens_initial_params)
    
#     if 'PEMD' in lens_model_list:
#         lens_params[2][0]['gamma'] = 2.
        
    lens_light_params[0] = deepcopy(kwargs_result['kwargs_lens_light'])                                

    source_params[0] = deepcopy(kwargs_result['kwargs_source'])                               

    lens_params[0] = deepcopy(kwargs_result['kwargs_lens'])
          
    (kwargs_params, lens_params,source_params,
     lens_light_params,ps_params,kwargs_result,chain_list,
     kwargs_likelihood, kwargs_model, kwargs_data_joint, 
     multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_lens, kwargs_psf,lens_model_list,
                                                 source_model_list,lens_light_model_list,lens_params, 
                                                 source_params, lens_light_params, 
                                                 kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, image_mask_list = mask_list) 
    
    with open(chainList_file, 'wb') as handle:
            pickle.dump(chain_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    lens_params[2] = deepcopy(lens_initial_params[2])
    
    kwargs_params = {'lens_model': deepcopy(lens_params),
                    'source_model': deepcopy(source_params),
                    'lens_light_model': deepcopy(lens_light_params)}
    
    kwargs_fixed = {'kwargs_lens': deepcopy(lens_params[2]), 
                     'kwargs_source': deepcopy(source_params[2]), 
                     'kwargs_lens_light': deepcopy(lens_light_params[2])}
    
    return (kwargs_params,kwargs_fixed, kwargs_result,chain_list,kwargs_likelihood, 
            kwargs_model, kwargs_data_joint, multi_band_list, kwargs_constraints)


def model_deblended_long(fitting_kwargs_list,lens_model_list,source_model_list,lens_light_model_list,
                        lens_initial_params,source_initial_params,lens_light_initial_params,
                        kwargs_data_lens,kwargs_data_LRG,kwargs_data_source,kwargs_psf,
                        num,object_ID,mask_list = None,kde_nsource=None,kde_Rsource=None,
                        source_mask_list = None, gal_mask_list =None,
                        fix_seed = False, fix_seed_val = None, 
                        save_seed_file = None, chainList_file = None,
                        results_path = None, band_list = None,):
    
    
    if fix_seed:
        np.random.set_state(fix_seed_val)
    else:
        np.random.seed(None)
        
    
    # get the initial state of the RNG
    save_seed_val = np.random.get_state()

    with open(save_seed_file, 'wb') as handle:
            pickle.dump(save_seed_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ########################################## Fit LRG image ##########################################
    print('I will first fit the LRG')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    # fit LRG:
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data_LRG)):
            f.append(deepcopy(lens_light_initial_params[j]))


    (kwargs_params, lens_params,source_params,
    lens_light_params,kwargs_result_LRG,chain_list,
    kwargs_likelihood, kwargs_model, kwargs_data_joint, 
    multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_LRG, kwargs_psf,
                                                  lens_model_list = [],
                                                  source_model_list = [],
                                                  lens_light_model_list = lens_light_model_list,
                                                  lens_params = lens_params, 
                                                  source_params = source_params, 
                                                  lens_light_params = lens_light_params, 
                                                  kde_nsource=kde_nsource,kde_Rsource=kde_Rsource,
                                                  image_mask_list = mask_list)
    
    
    
    
    LRG_path = results_path + '/LRG/'
    LRG_plot_path = LRG_path + 'modelPlots/'
    if not exists(LRG_path):
        os.mkdir(LRG_path)
    if not exists(LRG_plot_path):
        os.mkdir(LRG_plot_path)
    
    
    LRG_plot_kwargs = {'multi_band_list': multi_band_list, 'kwargs_model': kwargs_model,
                   'kwargs_params': kwargs_result_LRG, 'likelihood_mask_list': mask_list}
    
    
    LRG_plot_kwargs = plot_LRG_fit(LRG_plot_kwargs,band_list,LRG_plot_path,num, object_ID)
    
    with open(LRG_plot_path + 'Image_{}-{}_plot_kwargs.pickle'.format(num,object_ID),'wb') as handle:
        pickle.dump(LRG_plot_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ########################################## Fit lensed source image ##########################################
    print('I will now fit the lensed source')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    # fit lensed source
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for j,f in enumerate(source_params):
        for i in range(len(kwargs_data_source)):
            f.append(deepcopy(source_initial_params[j]))

    lens_params = deepcopy(lens_initial_params)
    
    
        

    (kwargs_params, lens_params,source_params,
    lens_light_params,kwargs_result_source,chain_list,
    kwargs_likelihood, kwargs_model, kwargs_data_joint, 
    multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_source, kwargs_psf,
                                                  lens_model_list = lens_model_list,
                                                  source_model_list = source_model_list,
                                                  lens_light_model_list = [],
                                                  lens_params = lens_params, 
                                                  source_params = source_params, 
                                                  lens_light_params = lens_light_params, 
                                                  kde_nsource=kde_nsource,kde_Rsource=kde_Rsource,
                                                  image_mask_list = source_mask_list)
    
    source_path = results_path + '/lensed_source/'
    source_plot_path = source_path + 'modelPlots/'
    if not exists(source_path):
        os.mkdir(source_path)
    if not exists(source_plot_path):
        os.mkdir(source_plot_path)
    
    
    source_plot_kwargs = {'multi_band_list': multi_band_list, 'kwargs_model': kwargs_model,
                   'kwargs_params': kwargs_result_source, 'likelihood_mask_list': source_mask_list}
    
    
    source_plot_kwargs = plot_lensed_source_fit(source_plot_kwargs,kwargs_data_source, kwargs_psf,
                           band_list,lens_model_list,source_model_list,lens_light_model_list,
                           source_plot_path,num, object_ID)
    
    with open(source_plot_path + 'Image_{}-{}_plot_kwargs.pickle'.format(num,object_ID),'wb') as handle:
        pickle.dump(source_plot_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    kwargs_result = deepcopy(kwargs_result_source)
    kwargs_result['kwargs_lens_light'] = deepcopy(kwargs_result_LRG['kwargs_lens_light'])
    
    
    
    ########################################## Optimize lens mass + source ##########################################
    print('I will now optimize the lensed source + lens mass of original image')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for l,x in enumerate(source_params):
        for i in range(len(kwargs_data_lens)):
            x.append(deepcopy(source_initial_params[l]))


    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data_lens)):
            f.append(deepcopy(lens_light_initial_params[j]))
            
    lens_params = deepcopy(lens_initial_params)
    
    lens_light_params[0] = deepcopy(kwargs_result['kwargs_lens_light'])                                

    source_params[0] = deepcopy(kwargs_result['kwargs_source'])                               

    lens_params[0] = deepcopy(kwargs_result['kwargs_lens'])
    
#     if 'PEMD' in lens_model_list:
#         lens_params[2][0]['gamma'] = 2.
        
    lens_light_params[2] = deepcopy(kwargs_result['kwargs_lens_light']) 

#     source_params[0] = deepcopy(kwargs_result['kwargs_source'])                               

#     lens_params[0] = deepcopy(kwargs_result['kwargs_lens'])
          
    (kwargs_params, lens_params,source_params,
     lens_light_params,kwargs_result,chain_list,
     kwargs_likelihood, kwargs_model, kwargs_data_joint, 
     multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_lens, kwargs_psf,lens_model_list,
                                                 source_model_list,lens_light_model_list,lens_params, 
                                                 source_params, lens_light_params, 
                                                 kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, image_mask_list = mask_list)
    
    ########################################## Optimize lens light ##########################################
    print('I will now optimize the lens light in the original image')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    
    
    lens_light_params = [[],[],[],[],[]]
   

    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data_lens)):
            f.append(deepcopy(lens_light_initial_params[j]))
    
    
    
    lens_light_params[0] = deepcopy(kwargs_result['kwargs_lens_light']) 
    

    source_params[0] = deepcopy(kwargs_result['kwargs_source'])                               

    lens_params[0] = deepcopy(kwargs_result['kwargs_lens'])        
    
#     if 'PEMD' in lens_model_list:
#         lens_params[2][0]['gamma'] = 2.
        


    source_params[2] = deepcopy(kwargs_result['kwargs_source'])                               

    lens_params[2] = deepcopy(kwargs_result['kwargs_lens'])
          
    (kwargs_params, lens_params,source_params,
     lens_light_params,kwargs_result,chain_list,
     kwargs_likelihood, kwargs_model, kwargs_data_joint, 
     multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_lens, kwargs_psf,lens_model_list,
                                                 source_model_list,lens_light_model_list,lens_params, 
                                                 source_params, lens_light_params, 
                                                 kde_nsource=kde_nsource,kde_Rsource=kde_Rsource,image_mask_list = mask_list) 
    
    ########################################## Optimize All ##########################################
    print('I will now fit the original image')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for l,x in enumerate(source_params):
        for i in range(len(kwargs_data_lens)):
            x.append(deepcopy(source_initial_params[l]))


    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data_lens)):
            f.append(deepcopy(lens_light_initial_params[j]))
            
    lens_params = deepcopy(lens_initial_params)
    
#     if 'PEMD' in lens_model_list:
#         lens_params[2][0]['gamma'] = 2.
        
    lens_light_params[0] = deepcopy(kwargs_result['kwargs_lens_light'])                                

    source_params[0] = deepcopy(kwargs_result['kwargs_source'])                               

    lens_params[0] = deepcopy(kwargs_result['kwargs_lens'])
          
    (kwargs_params, lens_params,source_params,
     lens_light_params,kwargs_result,chain_list,
     kwargs_likelihood, kwargs_model, kwargs_data_joint, 
     multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_lens, kwargs_psf,lens_model_list,
                                                 source_model_list,lens_light_model_list,lens_params, 
                                                 source_params, lens_light_params,
                                                 kde_nsource=kde_nsource,kde_Rsource=kde_Rsource, image_mask_list = mask_list) 
    
    with open(chainList_file, 'wb') as handle:
            pickle.dump(chain_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    lens_params[2] = deepcopy(lens_initial_params[2])
    
    kwargs_params = {'lens_model': deepcopy(lens_params),
                    'source_model': deepcopy(source_params),
                    'lens_light_model': deepcopy(lens_light_params)}
    
    kwargs_fixed = {'kwargs_lens': deepcopy(lens_params[2]), 
                     'kwargs_source': deepcopy(source_params[2]), 
                     'kwargs_lens_light': deepcopy(lens_light_params[2])}
    
    return (kwargs_params,kwargs_fixed, kwargs_result,chain_list,kwargs_likelihood, 
            kwargs_model, kwargs_data_joint, multi_band_list, kwargs_constraints)





def model_deblended_LRG(fitting_kwargs_list,lens_model_list,source_model_list,lens_light_model_list,
                        lens_initial_params,source_initial_params,lens_light_initial_params,
                        kwargs_data_lens,kwargs_data_LRG,kwargs_data_source,kwargs_psf,mask_list):
    
    ########################################## Optimize lens light ##########################################
    print('I will first optimize the lens light using the deblended LRG image')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    # fit LRG:
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data_LRG)):
            f.append(deepcopy(lens_light_initial_params[j]))


    (kwargs_params, lens_params,source_params,
    lens_light_params,kwargs_result_LRG,chain_list,
    kwargs_likelihood, kwargs_model, kwargs_data_joint, 
    multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_LRG, kwargs_psf,
                                                  lens_model_list = [],
                                                  source_model_list = [],
                                                  lens_light_model_list = lens_light_model_list,
                                                  lens_params = lens_params, 
                                                  source_params = source_params, 
                                                  lens_light_params = lens_light_params,
                                                  image_mask_list = mask_list)
    

    
    
#     kwargs_result = deepcopy(kwargs_result_source)
#     kwargs_result['kwargs_lens_light'] = deepcopy(kwargs_result_LRG['kwargs_lens_light'])
    
    ########################################## Optimize lens mass + source ##########################################
    print('I will now optimize the lensed source + lens mass of original image')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for l,x in enumerate(source_params):
        for i in range(len(kwargs_data_lens)):
            x.append(deepcopy(source_initial_params[l]))


    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data_lens)):
            f.append(deepcopy(lens_light_initial_params[j]))
            
    lens_params = deepcopy(lens_initial_params)
    
#     if 'PEMD' in lens_model_list:
#         lens_params[2][0]['gamma'] = 2.
        
    lens_light_params[0] = deepcopy(kwargs_result_LRG['kwargs_lens_light']) 
    lens_light_params[2] = deepcopy(kwargs_result_LRG['kwargs_lens_light']) 

#     source_params[0] = deepcopy(kwargs_result['kwargs_source'])                               

#     lens_params[0] = deepcopy(kwargs_result['kwargs_lens'])
          
    (kwargs_params, lens_params,source_params,
     lens_light_params,kwargs_result,chain_list,
     kwargs_likelihood, kwargs_model, kwargs_data_joint, 
     multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_lens, kwargs_psf,lens_model_list,
                                                 source_model_list,lens_light_model_list,lens_params, 
                                                 source_params, lens_light_params, image_mask_list = mask_list) 
    
    
   ########################################## Optimize lens light ##########################################
    print('I will now optimize the lens light in the original image')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    
    
    lens_light_params = [[],[],[],[],[]]
   

    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data_lens)):
            f.append(deepcopy(lens_light_initial_params[j]))
    
    
    
    lens_light_params[0] = deepcopy(kwargs_result['kwargs_lens_light']) 
    

    source_params[0] = deepcopy(kwargs_result['kwargs_source'])                               

    lens_params[0] = deepcopy(kwargs_result['kwargs_lens'])        
    
#     if 'PEMD' in lens_model_list:
#         lens_params[2][0]['gamma'] = 2.
        


    source_params[2] = deepcopy(kwargs_result['kwargs_source'])                               

    lens_params[2] = deepcopy(kwargs_result['kwargs_lens'])
          
    (kwargs_params, lens_params,source_params,
     lens_light_params,kwargs_result,chain_list,
     kwargs_likelihood, kwargs_model, kwargs_data_joint, 
     multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_lens, kwargs_psf,lens_model_list,
                                                 source_model_list,lens_light_model_list,lens_params, 
                                                 source_params, lens_light_params, image_mask_list = mask_list)       
    
    
    ########################################## Optimize All ##########################################
    print('I will now optimize Everything')
    print('\n')
    print('-------------------------------------------------------------------')
    print('\n')
    
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]

    for l,x in enumerate(source_params):
        for i in range(len(kwargs_data_lens)):
            x.append(deepcopy(source_initial_params[l]))


    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data_lens)):
            f.append(deepcopy(lens_light_initial_params[j]))
            
    lens_params = deepcopy(lens_initial_params)
    
#     if 'PEMD' in lens_model_list:
#         lens_params[2][0]['gamma'] = 2.
        
    lens_light_params[0] = deepcopy(kwargs_result['kwargs_lens_light'])                                

    source_params[0] = deepcopy(kwargs_result['kwargs_source'])                               

    lens_params[0] = deepcopy(kwargs_result['kwargs_lens'])
          
    (kwargs_params, lens_params,source_params,
     lens_light_params,kwargs_result,chain_list,
     kwargs_likelihood, kwargs_model, kwargs_data_joint, 
     multi_band_list,kwargs_constraints) = run_fit(fitting_kwargs_list,kwargs_data_lens, kwargs_psf,lens_model_list,
                                                 source_model_list,lens_light_model_list,lens_params, 
                                                 source_params, lens_light_params, image_mask_list = mask_list) 
    
    
    
    
    kwargs_params = {'lens_model': deepcopy(lens_params),
                    'source_model': deepcopy(source_params),
                    'lens_light_model': deepcopy(lens_light_params)}
    
    kwargs_fixed = {'kwargs_lens': deepcopy(lens_params[2]), 
                     'kwargs_source': deepcopy(source_params[2]), 
                     'kwargs_lens_light': deepcopy(lens_light_params[2])}
    
    return (kwargs_params,kwargs_fixed, kwargs_result,chain_list,kwargs_likelihood, 
            kwargs_model, kwargs_data_joint, multi_band_list, kwargs_constraints)



#################################################################################################

######################################### Full Sampling #########################################

#################################################################################################


def full_sampling(fitting_kwargs_list,kwargs_params, kwargs_data, kwargs_psf,
                  lens_model_list, source_model_list,lens_light_model_list,
                  ps_model_list = None,
                  kde_nsource=None,kde_Rsource=None,mask_list=None):
    
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
    kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list, kwargs_constraints = prepareFit(kwargs_data,
                                                                                              kwargs_psf,lens_model_list, 
                                                                                              source_model_list,
                                                                                              lens_light_model_list,
                                                                                              ps_model_list = ps_model_list,
                                                                                              kde_nsource=kde_nsource,
                                                                                              kde_Rsource=kde_Rsource, 
                                                                                              check_pos_flux = True,
                                                                                              image_mask_list = mask_list)
    
    chain_list, kwargs_result = runFit(fitting_kwargs_list, kwargs_params, 
                                       kwargs_likelihood, kwargs_model,
                                       kwargs_data_joint, kwargs_constraints = kwargs_constraints)  
    
    kwargs_params['lens_model'][0] = deepcopy(kwargs_result['kwargs_lens'])
    kwargs_params['source_model'][0] = deepcopy(kwargs_result['kwargs_source'])
    kwargs_params['lens_light_model'][0] = deepcopy(kwargs_result['kwargs_lens_light'])
    if ps_model_list != None:
        kwargs_params['point_source_model'][0] = deepcopy(kwargs_result['kwargs_ps'])
    
    
    
    return (chain_list, kwargs_result,kwargs_params,
            kwargs_likelihood, kwargs_model, kwargs_data_joint, 
            multi_band_list, kwargs_constraints)


#################################################################################################

############################################# Misc. #############################################

#################################################################################################



def removekeys(d, keys):
    ''' 
    Create new dictionary, r, out of dictionary d with keys removed. 
    '''
    r = dict(d)
    for k in keys:
        del r[k]
    return r

def optParams(kwargs_init,opt_params,model_kwarg_names):
    '''
    Function for setting up init and fixed dictionaries for optimizing specific model parameters and fixing the rest. Can be 
    used for making a chain of PSO fits in pre-optimization scheme as desired in a less messy way. Used in optimize_dynamic.py
    script.
    
        :kwargs_init: initial parameter values for modeling. Same format as kwargs_result after a fit. 
        :opt_params: dictionary with 'kwargs_lens','kwargs_source', and 'kwargs_lens_light' as keys. Values are arrays with names 
        of parameters to optimize in the next fit. 
        :model_kwarg_names: Dictionary. All parameter names in profiles in source, lens, and lens_light model lists.
    
    Returns: 
        :args_init: dictionary of initialization parameters
        :args_fixed: dictionary of fixed parametes
    '''
    
    fixed_lens = []
    kwargs_lens_init = []
    
    for i in range(len(model_kwarg_names['kwargs_lens'])):     # num iterations correspond to num profiles in lens_model_list
        kwargs_lens_init.append(kwargs_init['kwargs_lens'][i])  
        
        #fix all params (and their values) that are not being optimized
        fixed_lens.append(removekeys(kwargs_init['kwargs_lens'][i],opt_params['kwargs_lens'][i]))  
    
    fixed_source = []
    kwargs_source_init = []
    
    for i in range(len(model_kwarg_names['kwargs_source'])):
        kwargs_source_init.append(kwargs_init['kwargs_source'][i])
        fixed_source.append(removekeys(kwargs_init['kwargs_source'][i],opt_params['kwargs_source'][i]))
    
    fixed_lens_light = []
    kwargs_lens_light_init = []
    
    for i in range(len(model_kwarg_names['kwargs_lens_light'])):
        kwargs_lens_light_init.append(kwargs_init['kwargs_lens_light'][i])
        fixed_lens_light.append(removekeys(kwargs_init['kwargs_lens_light'][i],opt_params['kwargs_lens_light'][i]))
    
    args_init = {'kwargs_lens': kwargs_lens_init, 
                 'kwargs_source': kwargs_source_init, 
                 'kwargs_lens_light': kwargs_lens_light_init}
    
    args_fixed = {'kwargs_lens': fixed_lens, 
                 'kwargs_source': fixed_source, 
                 'kwargs_lens_light': fixed_lens_light}
    return args_init, args_fixed

