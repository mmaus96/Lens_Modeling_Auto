from Lens_Modeling_Auto.auto_modeling_functions import prepareFit
from Lens_Modeling_Auto.auto_modeling_functions import runFit
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from Lens_Modeling_Auto.auto_modeling_functions import find_components
from Lens_Modeling_Auto.auto_modeling_functions import mask_for_sat
from Lens_Modeling_Auto.auto_modeling_functions import mask_for_lens_gal


####################### Initial Params #######################
lens_light_sersic_fixed = {}
lens_light_sersic_init = {'R_sersic': 1.0, 'n_sersic': 2.,'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0}
# lens_light_sersic_init = {'R_sersic': 2.0, 'n_sersic': 2.5,'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0}
lens_light_sersic_sigma = {'R_sersic': 0.1, 'n_sersic': 1.0,'e1': 0.5, 'e2': 0.5,  'center_x': 0.01, 'center_y': 0.01}
lens_light_sersic_lower = {'R_sersic': 0.001, 'n_sersic': 1.0,'e1': -0.5, 'e2': -0.5, 'center_x': -1.5, 'center_y': -1.5}
lens_light_sersic_upper = {'R_sersic': 5., 'n_sersic': 10.,'e1': 0.5, 'e2': 0.5, 'center_x': 1.5, 'center_y': 1.5}

source_sersic_fixed = {}
source_sersic_init = {'R_sersic': 1.0, 'n_sersic': 1.,'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0}
# source_sersic_init = {'R_sersic': 0.5, 'n_sersic': 2.5,'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0}
source_sersic_sigma = {'R_sersic': 0.1, 'n_sersic': 0.5,'e1': 0.5, 'e2': 0.5,  'center_x': 0.01, 'center_y': 0.01}
source_sersic_lower = {'R_sersic': 0.001, 'n_sersic': 0.1,'e1': -0.5, 'e2': -0.5, 'center_x': -1.5, 'center_y': -1.5}
source_sersic_upper = {'R_sersic': 5., 'n_sersic': 10.,'e1': 0.5, 'e2': 0.5, 'center_x': 1.5, 'center_y': 1.5}


lens_sie_fixed = {}  
lens_sie_init = {'theta_E': 1.5, 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.}
lens_sie_sigma = {'theta_E': .3, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.1, 'center_y': 0.1}
lens_sie_lower = {'theta_E': 0.1, 'e1': -1, 'e2': -1, 'center_x': -1.5, 'center_y': -1.5}
lens_sie_upper = {'theta_E': 5.0, 'e1': 1, 'e2': 1, 'center_x': 1.5, 'center_y': 1.5}


lens_shear_fixed = {'ra_0': 0, 'dec_0': 0}
lens_shear_init = {'ra_0': 0, 'dec_0': 0,'gamma1': 0., 'gamma2': 0.0}
lens_shear_sigma = {'gamma1': 0.1, 'gamma2': 0.1}
lens_shear_lower = {'gamma1': -0.5, 'gamma2': -0.5}
lens_shear_upper = {'gamma1': 0.5, 'gamma2': 0.5}

if includeShear == True:
    lens_initial_params = deepcopy([[lens_sie_init,lens_shear_init],
                                    [lens_sie_sigma,lens_shear_sigma],
                                    [lens_sie_fixed,lens_shear_fixed],
                                    [lens_sie_lower,lens_shear_lower],
                                    [lens_sie_upper,lens_shear_upper]])
else:
    lens_initial_params = deepcopy([[lens_sie_init],
                                    [lens_sie_sigma],
                                    [lens_sie_fixed],
                                    [lens_sie_lower],
                                    [lens_sie_upper]])


lens_light_initial_params = deepcopy([[lens_light_sersic_init], 
                                      [lens_light_sersic_sigma], 
                                      [lens_light_sersic_fixed], 
                                      [lens_light_sersic_lower], 
                                      [lens_light_sersic_upper]])

source_initial_params = deepcopy([[source_sersic_init], 
                                  [source_sersic_sigma], 
                                  [source_sersic_fixed], 
                                  [source_sersic_lower], 
                                  [source_sersic_upper]])



########################################## Everything ##########################################


print('I will now Fit Everything')
print('\n')
print('-------------------------------------------------------------------')
print('\n')

#Model Lists
if includeShear == True:
    lens_model_list = ['SIE','SHEAR'] 
else:
    lens_model_list = ['SIE']
    
source_model_list = ['SERSIC_ELLIPSE']
lens_light_model_list = ['SERSIC_ELLIPSE']

# gal_mask_list = []
# mask_list = []

# for data in kwargs_data: 
#     gal_mask_list.append(mask_for_lens_gal(data['image_data'],deltaPix))
#     if use_mask:
#         mask_list.append(mask_for_sat(data['image_data'],deltaPix))
#     else: mask_list = None


#prepare fitting kwargs
kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list,kwargs_constraints = prepareFit(kwargs_data, kwargs_psf,
                                                                                 lens_model_list, source_model_list,
                                                                                 lens_light_model_list, 
                                                                                 image_mask_list = mask_list)  

#prepare kwarg_params
lens_light_params = [[],[],[],[],[]]
source_params = [[],[],[],[],[]]
lens_params = [[],[],[],[],[]]

for l,x in enumerate(source_params):
    for i in range(len(kwargs_data)):
        x.extend(deepcopy(source_initial_params[l]))
        
        
for j,f in enumerate(lens_light_params):
    for i in range(len(kwargs_data)):
        f.extend(deepcopy(lens_light_initial_params[j]))


lens_params = deepcopy(lens_initial_params)

# lens_initial_params = deepcopy(lens_params)

kwargs_params = {'lens_model': deepcopy(lens_params),
                'source_model': deepcopy(source_params),
                'lens_light_model': deepcopy(lens_light_params)}

kwargs_fixed = {'kwargs_lens': deepcopy(lens_params[2]), 
                 'kwargs_source': deepcopy(source_params[2]), 
                 'kwargs_lens_light': deepcopy(lens_light_params[2])}


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

kwargs_fixed = {'kwargs_lens': deepcopy(lens_params[2]), 
                 'kwargs_source': deepcopy(source_params[2]), 
                 'kwargs_lens_light': deepcopy(lens_light_params[2])}


print('\n')
print('##########################################################################')
print('\n')


########################################## Optimize lens light ##########################################
print('I will now optimize lens light')
print('\n')
print('-------------------------------------------------------------------')
print('\n')



#prepare fitting kwargs
kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list,kwargs_constraints = prepareFit(kwargs_data, kwargs_psf,
                                                                                 lens_model_list, source_model_list,
                                                                                 lens_light_model_list, 
                                                                                 image_mask_list = mask_list)  

#prepare kwarg_params

lens_light_params[0] = deepcopy(kwargs_result['kwargs_lens_light'])                                

source_params[2] = deepcopy(kwargs_result['kwargs_source'])                               

lens_params[2] = deepcopy(kwargs_result['kwargs_lens']) 




# lens_initial_params = deepcopy(lens_params)

kwargs_params = {'lens_model': deepcopy(lens_params),
                'source_model': deepcopy(source_params),
                'lens_light_model': deepcopy(lens_light_params)}

kwargs_fixed = {'kwargs_lens': deepcopy(lens_params[2]), 
                 'kwargs_source': deepcopy(source_params[2]), 
                 'kwargs_lens_light': deepcopy(lens_light_params[2])}


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
########################################## Model Lens and Source Again ##########################################
print('\n')
print('##########################################################################')
print('\n')
print('I will now optimize the source and lens profiles')
print('\n')
print('-------------------------------------------------------------------')
print('\n')

#Fix Lens Light
lens_light_params[2] = deepcopy(kwargs_result['kwargs_lens_light']) 
source_params[2] = []
for i in range(len(kwargs_data)):
    source_params[2].extend(deepcopy(source_initial_params[2]))
lens_params[2] = deepcopy(lens_initial_params[2])


kwargs_params = {'lens_model': deepcopy(lens_params),
                'source_model': deepcopy(source_params),
                'lens_light_model': deepcopy(lens_light_params)}

#prepare fitting kwargs
kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list,kwargs_constraints = prepareFit(kwargs_data, kwargs_psf,
                                                                                 lens_model_list, source_model_list,
                                                                                 lens_light_model_list, 
                                                                                 image_mask_list = mask_list)  
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


lens_light_params[2] = []
source_params[2] = []
for i in range(len(kwargs_data)):
    source_params[2].extend(deepcopy(source_initial_params[2]))
    lens_light_params[2].extend(deepcopy(lens_light_initial_params[2]))
lens_params[2] = deepcopy(lens_initial_params[2])

lens_light_params[0] = deepcopy(kwargs_result['kwargs_lens_light'])                                

source_params[0] = deepcopy(kwargs_result['kwargs_source'])                               

lens_params[0] = deepcopy(kwargs_result['kwargs_lens']) 

kwargs_params = {'lens_model': deepcopy(lens_params),
                'source_model': deepcopy(source_params),
                'lens_light_model': deepcopy(lens_light_params)}

kwargs_fixed = {'kwargs_lens': deepcopy(lens_params[2]), 
                 'kwargs_source': deepcopy(source_params[2]), 
                 'kwargs_lens_light': deepcopy(lens_light_params[2])}



multi_source_model_list = []
multi_lens_light_model_list = []
    
for i in range(len(kwargs_data)):
    multi_source_model_list.extend(deepcopy(source_model_list))
    multi_lens_light_model_list.extend(deepcopy(lens_light_model_list))

################## Write Initial Params to text file ##################

file = open(results_path+"/initial_params.txt","a")#append mode 
file.write("Model lists: \n")
file.write("lens model: " + str(lens_model_list) + " \n")
file.write("source model: " + str(multi_source_model_list) + " \n")
file.write("lens light model: "+ str(multi_lens_light_model_list) + " \n")

file.write("\n") 
file.write("\n")
file.write("kwargs_lens (init,sigma,fixed,lower,upper): \n") 
for l in lens_initial_params:
    file.write(str(l)+ " \n") 

file.write("\n")
file.write("kwargs_source (init,sigma,fixed,lower,upper): \n") 

for s in source_initial_params:
    file.write(str(s) + " \n") 

file.write("\n")    
file.write("kwargs_lens_light (init,sigma,fixed,lower,upper): \n") 
for l in lens_light_initial_params:
    file.writelines(str(l)+ " \n") 
    
file.write("\n") 
file.write("\n") 

file.write("kwargs_model: \n") 
print(kwargs_model, file = file)
file.write("\n") 
file.write("\n")

file.write("\n")
file.write("kwargs_constraints: \n") 
print(kwargs_constraints, file = file)
file.write("\n") 
file.write("\n")

file.write("\n")
file.write("kwargs_likelihood: \n") 
print(kwargs_likelihood, file = file)
file.write("\n") 
file.write("\n")

file.write('\n')
file.write('\n'+'Custom_LogL_Priors: ' + str([[0,'q',0.8,0.1]]) + ' \n')
file.write('\n')
    
file.close() 
