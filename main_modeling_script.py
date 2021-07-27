import sys
if sys.platform[:5] == 'linux':
    import matplotlib
    matplotlib.use('Agg')

import re
import os, psutil
from os import walk
from os import listdir
from os.path import isfile, join, isdir, exists
import time
import numpy as np
import pandas as pd
from copy import deepcopy
import lenstronomy
import astropy
import scipy
import pickle
from Lens_Modeling_Auto.auto_modeling_functions import openFITS
from Lens_Modeling_Auto.auto_modeling_functions import calcBackgroundRMS
from Lens_Modeling_Auto.auto_modeling_functions import prepareData
from Lens_Modeling_Auto.auto_modeling_functions import get_kwarg_names
from Lens_Modeling_Auto.auto_modeling_functions import printMemory
from Lens_Modeling_Auto.auto_modeling_functions import mask_for_sat
from Lens_Modeling_Auto.auto_modeling_functions import estimate_radius
from Lens_Modeling_Auto.auto_modeling_functions import find_lens_gal
from Lens_Modeling_Auto.fit_sequence_functions import initial_model_params
from Lens_Modeling_Auto.fit_sequence_functions import initial_modeling_fit
from Lens_Modeling_Auto.fit_sequence_functions import full_sampling
from Lens_Modeling_Auto.plot_functions import make_modelPlots
from Lens_Modeling_Auto.plot_functions import make_chainPlots
from Lens_Modeling_Auto.plot_functions import make_cornerPlots
from Lens_Modeling_Auto.plot_functions import save_chain_list




#####################################################################################################################

#################################################### User Inputs ####################################################

#####################################################################################################################

# nohup python -u ./Lens_Modeling_Auto/main_modeling_script.py > lens_candidates/Group1/PEMD_lens/results_Mar29/output_logs/output0_6.log &

# print('lenstronomy version: {}'.format(lenstronomy.__version__))

# file paths to image data and results destination [TO DO BY USER]
data_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1'
results_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/test_save_mask'
# results_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/PEMD_lens/results_Mar29'

# data_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Sure_Lens'
# results_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Sure_Lens/results_e1e2_free'

if not exists(results_path):
    os.mkdir(results_path)

#Folder names for data, psf, noise map, original image [TO DO BY USER]
im_path = data_path + '/data'
# im_path = data_path + '/simulations'
psf_path = data_path + '/psf'
noise_path = data_path + '/psf'
noise_type = 'EXPTIME'
band_list = ['g','r','i']
obj_name_location = 0

#Modeling Options [TO DO BY USER]
use_shapelets = False
use_mask = True
lens_model_list = ['SIE','SHEAR']
source_model_list = ['SERSIC_ELLIPSE']
lens_light_model_list = ['SERSIC_ELLIPSE']
this_is_a_test = True
numCores = 1 

select_objects = None #['06653211','06788344','07087938','11308901','25315733','19990514']  #List of strings with object IDs, or None

# Additional info for images [TO DO BY USER]
deltaPix = 0.27
zeroPt = 30
psf_upsample_factor = 1
ra_dec = 'csv' # 'csv', 'header', or 'None'
ra_dec_loc = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/group_1.csv' #path to csv file or header file, or 'None'
Mask_rad_file = None #'/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/mask_v2.csv' #path to csv file or 'None'

id_col_name = 'id_1'

printMemory('Beginning')

#####################################################################################################################

########################################### Find Data and sort filenames ############################################

#####################################################################################################################



im_files = [f for f in listdir(im_path) if isfile('/'.join([im_path,f]))]
psf_files,noise_files = [],[]
psf_files_dict, noise_files_dict = {},{}

for b in band_list: 
    psf_files.append([f for f in listdir(psf_path + '/' + b) if isfile('/'.join([psf_path + '/' + b,f]))])
    noise_files.append([f for f in listdir(noise_path + '/' + b) if isfile('/'.join([noise_path + '/' + b,f]))])
    psf_files_dict[b] = [f for f in listdir(psf_path + '/' + b) if isfile('/'.join([psf_path + '/' + b,f]))]
    noise_files_dict[b] = [f for f in listdir(noise_path + '/' + b) if isfile('/'.join([noise_path + '/' + b,f]))]



    
obj_names = []
if not select_objects:
    for x in im_files:
        obj_names.append(re.findall('\d+', x)[obj_name_location])
else: obj_names = deepcopy(select_objects)


data_pairs_dicts = []
for i,obj in enumerate(obj_names):
    for x in im_files:
        if obj in x: im = x
            
    psf = {}
    for b in band_list:
        for file in psf_files_dict[b]:
            if obj in file: psf[b] = '/'.join([b,file])
    
    noise = {}
    for b in band_list:
        for file in noise_files_dict[b]:
            if obj in file: noise[b]= '/'.join([b,file])

    if ra_dec == 'csv':
        df_info = pd.read_csv(ra_dec_loc)
        RA, DEC = 'N/A','N/A'
        for j in range(len(df_info.loc[:,:])):
            if int(df_info.loc[j,id_col_name]) == int(obj): RA,DEC = df_info.loc[j,'ra'],df_info.loc[j,'dec']
    else: RA, DEC = 'N/A','N/A'
    
    data_pairs_dicts.append({'image_data': im , 'psf': psf , 'noise_map': noise, 
                             'noise_type': noise_type,'object_ID': obj,'RA': RA, 'DEC': DEC})

data_pairs_dicts = sorted(data_pairs_dicts, key=lambda k: float(k['object_ID']))
data_pairs_cut = []
print('\n')
print('############## Files Organized #################')
print('files to model:')
print('\n')
count = 0
for i,x in enumerate(data_pairs_dicts): 
    if (not x['psf']) or (not x['noise_map']):
        continue
    count += 1
    print('image {}'.format(count))
    print('ID: {}'.format(x['object_ID']))
    print('RA: {}, DEC: {}'.format(x['RA'],x['DEC']))
    print('Image data: ',x['image_data'])
    print('PSF: ',x['psf'])
    print('Noise: ',x['noise_map'])
    print('\n')
    data_pairs_cut.append(x)
    
data_pairs_dicts = deepcopy(data_pairs_cut)
print('\n')
print('I will now begin modeling the images')
print('\n')   
    
#####################################################################################################################     

################################################### Begin Modeling ##################################################

#####################################################################################################################

f = open(results_path + "/initial_params.txt","w")#append mode
f.write('\n' + '###############################################################################################' + ' \n')
f.write('\n')
f.write('\n' + '################################### Modeling Initial Params ###################################' + ' \n')
f.write('\n')
f.write('\n' + '###############################################################################################' + ' \n')
f.write('\n')
f.write('lenstronomy version: {}'.format(lenstronomy.__version__))
f.write('\n')
f.write('numpy version: {}'.format(np.__version__))
f.write('\n')
f.write('astropy version: {}'.format(astropy.__version__))
f.write('\n')
f.write('scipy version: {}'.format(scipy.__version__))
f.write('\n')
f.close()

printMemory('Before loop')
    
for it in range(len(data_pairs_dicts)):    
#     it += 42 
    
#     if (not data_pairs_dicts[it]['psf']) or (not data_pairs_dicts[it]['noise_map']):
#         continue
#     elif (good_images_indices != None) and (it not in good_images_indices):
#         continue
    
    
    print('\n')
    print('modeling image {}'.format(it + 1))
    print('\n')
    print(data_pairs_dicts[it])
    print('\n')
    tic = time.perf_counter()
    
    f = open(results_path + "/initial_params.txt","a")#append mode
    f.write('\n')
    f.write('\n' + '################################### image {} ###################################'.format(it + 1) + ' \n')
    f.write('\n')
    print(data_pairs_dicts[it],file = f)
    f.write('\n')
    f.close()
    
    #band_index = np.where(np.array(band_list) == band)[0][0]
    data,hdr = openFITS(im_path + '/' + data_pairs_dicts[it]['image_data'])
    psf, psf_hdr = [],[]
    noise_map,noise_hdr = [],[]
    for b in band_list:
        d,h = openFITS(psf_path  + '/' + data_pairs_dicts[it]['psf'][b])
        if np.ndim(d)== 3:
            psf.extend(d)
        elif np.ndim(d)== 2:
            psf.append(d)
        psf_hdr.append(h)
        
    #     psf.extend(d)
    #     psf_hdr.extend(h)
#         psf.append(d)
#         psf_hdr.append(h)


        d2,h2 = openFITS(noise_path  + '/' + data_pairs_dicts[it]['noise_map'][b])
        if np.ndim(d2)== 3:
            noise_map.extend(d2)
        elif np.ndim(d2)== 2:
            noise_map.append(d2)
        noise_hdr.append(h2)
    
    #     noise_map.extend(d2)
    #     noise_hdr.extend(h2)
#         noise_map.append(d2)
#         noise_hdr.append(h2)


    data_dict = {'image_data': [], 'image_hdr': [], 
                 'psf': psf, 'psf_hdr': psf_hdr, 
                 'noise_map': noise_map, 'noise_hdr': noise_hdr}
    
    printMemory('After openFITS')

    for i,b in enumerate(band_list):
#         for j,h in enumerate(hdr):
#             if h['BAND'] == b:
#         data_dict['image_data'].append(data[i])
#         data_dict['image_hdr'].append(hdr[0])
        if np.ndim(data) == 4:
            data_dict['image_data'].append(data[0][i])
        elif np.ndim(data) == 3:
            data_dict['image_data'].append(data[i])
        data_dict['image_hdr'].append(hdr[0])
    
    print('calculating background values')
    print('\n')
    background_rms = calcBackgroundRMS(data_dict['image_data']) #calculate rms background
    print('\n')

    lens_info = []

    for i,x in enumerate(data_dict['image_data']):

        lens_info.append({'deltaPix': deltaPix ,
                         'numPix': len(x),
                         'background_rms': background_rms[i],
                         'psf_type': 'PIXEL',
                         'psf_upsample_factor': psf_upsample_factor})

        if noise_type == 'EXPTIME': 
            lens_info[i]['exposure_time'] = data_dict['noise_hdr'][i][0]['EXPTIME']
            lens_info[i]['noise_map'] = None
        else:
            lens_info[i]['exposure_time'] = None
            lens_info[i]['noise_map'] = data_dict['noise_map'][i]

    kwargs_data, kwargs_psf = prepareData(lens_info,data_dict['image_data'],
                                           data_dict['psf']) 
    
    
    printMemory('After prepareData')
    
    
    ############################## Prepare Mask ############################
    
    c_x,c_y = find_lens_gal(kwargs_data[-1]['image_data'],deltaPix,show_plot=True,title=data_pairs_dicts[it]['object_ID'])
    
    if Mask_rad_file == None:
        mask_size_ratio = None
        mask_size_px,mask_size_as = estimate_radius(kwargs_data[0]['image_data'],
                                                    deltaPix,center_x=c_x,center_y=c_y,show_plot=False, name = None)
    else:
        df_mask = pd.read_csv(Mask_rad_file)
        mask_size_ratio = None
        mask_size_as = float(df_mask.loc[df_mask[id_col_name] == int(data_pairs_dicts[it]['object_ID']),'dst_arcsec']) #+8.*deltaPix 
        
    
    gal_mask_list = []
    mask_list = []
#     sizes_As = []
#     sizes_px = []

    if use_mask:
        if not exists(results_path + '/masks'):
            os.mkdir(results_path + '/masks')
        mask_path = results_path + '/masks'
        for k,data in enumerate(kwargs_data): 
            
            band_path = mask_path + '/' + band_list[k]
            if not exists(band_path):
                os.mkdir(band_path)

            mask = mask_for_sat(data['image_data'],deltaPix,
                                          lens_rad_arcsec = mask_size_as,
                                          center_x=c_x,center_y=c_y,
                                          lens_rad_ratio = mask_size_ratio, 
                                          show_plot = False)
            mask_list.append(mask)

            with open(band_path + '/Image_{}-{}.pickle'.format(it+1, data_pairs_dicts[it]['object_ID']), 'wb') as handle:
                    pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #             sizes_As.append(size_Arcsec)
    #             sizes_px.append(size_pix)
        
    else: mask_list = None
    
    file = open(results_path+"/initial_params.txt","a")#append mode 
    file.write("Mask Size: \n")
    file.write("{} pixels,{} arcsec \n".format(mask_size_px,mask_size_as))
    file.write("Mask Center: \n")
    file.write("({},{}) \n".format(c_x,c_y))
    file.close() 
    
    ################################################################################################################# 
    
    ################################################## Initial PSOs #################################################
    
    ################################################################################################################# 
    print('\n')
    print('I will start with initial fits of the lens, source and lens light profiles')
    print('\n')
    
    
    if this_is_a_test:
        fitting_kwargs_list = [['PSO', {'sigma_scale': 1, 'n_particles': 50, 'n_iterations': 50,'threadCount': numCores}]
#                                 ,['MCMC', {'n_burn': 0, 'n_run': 50, 'walkerRatio': 10, 'sigma_scale': .1,'threadCount':numCores}]
                              ]
    else:
        fitting_kwargs_list = [['PSO', {'sigma_scale': 1, 'n_particles': 100, 'n_iterations': 2000,'threadCount': numCores}]
                                #,['MCMC', {'n_burn': 0, 'n_run': 100, 'walkerRatio': 10, 'sigma_scale': .1,'threadCount':numCores}]
                              ]

   
    
    
    lens_initial_params,source_initial_params,lens_light_initial_params = initial_model_params(lens_model_list)
    
    kwargs_params,kwargs_fixed, kwargs_result,chain_list,kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list, kwargs_constraints = initial_modeling_fit(fitting_kwargs_list,lens_model_list,source_model_list,
                                          lens_light_model_list,lens_initial_params,
                                          source_initial_params,lens_light_initial_params,
                                          kwargs_data,kwargs_psf,mask_list)

#     exec(open('Lens_Modeling_Auto/initial_modeling_fit.py').read())
    
    printMemory('After initial fit')
    
    
    toc1 = time.perf_counter()                
    
    print('\n')
    print('First sampling took: {:.2f} minutes'.format((toc1 - tic)/60.0))
    
    
    multi_source_model_list = []
    multi_lens_light_model_list = []
    
    for i in range(len(kwargs_data)):
        multi_source_model_list.extend(deepcopy(source_model_list))
        multi_lens_light_model_list.extend(deepcopy(lens_light_model_list))
        
    model_kwarg_names = get_kwarg_names(lens_model_list,multi_source_model_list,
                                         multi_lens_light_model_list,kwargs_fixed)
    
    
    ################################################################################################################# 
    
    ################################################# Full Sampling #################################################
    
    ################################################################################################################# 
   
    print('\n')
    print('I will now run the full sampling')
    print('\n')
    
    if this_is_a_test:
        fitting_kwargs_list = [['PSO', {'sigma_scale': 0.1, 'n_particles': 50, 'n_iterations': 50,'threadCount': numCores}]
#                                 ,['MCMC', {'n_burn': 0, 'n_run': 50, 'walkerRatio': 10, 'sigma_scale': .05,'threadCount':numCores}]
                              ]
    else:
        fitting_kwargs_list = [['PSO', {'sigma_scale': 1, 'n_particles': 150, 'n_iterations': 2000,'threadCount': numCores}]
                                ,['MCMC', {'n_burn': 200, 'n_run': 1000, 'walkerRatio': 10, 'sigma_scale': .05,'threadCount':numCores}]
                              ]
    
    
    
    chain_list, kwargs_result,kwargs_params,kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list, kwargs_constraints = full_sampling(fitting_kwargs_list,kwargs_params, 
                                                        kwargs_data, kwargs_psf,lens_model_list, 
                                                        source_model_list,lens_light_model_list,
                                                        mask_list)
    
#     if not this_is_a_test:
#         exec(open('Lens_Modeling_Auto/Full_Sampling.py').read())
    
    printMemory('After Full Sampling')

    toc2 = time.perf_counter()
    print('\n')
    print('Full sampling took: {:.2f} minutes'.format((toc2 - toc1)/60.0), '\n',
         'Total time: {:.2f} minutes'.format((toc2 - tic)/60.0))
    
    
    print('\n')
    
    ################################################################################################################# 
    
    ######################################### Create Plots and Save Results #########################################
    
    ################################################################################################################# 
    
#     if it == 0:
    if not exists(results_path + '/modelPlot_results'):
        os.mkdir(results_path + '/modelPlot_results')
    if not exists(results_path + '/chainPlot_results'):
        os.mkdir(results_path + '/chainPlot_results')
    if not exists(results_path + '/cornerPlot_results'):
        os.mkdir(results_path + '/cornerPlot_results')
    if not exists(results_path + '/chain_lists'):
        os.mkdir(results_path + '/chain_lists')
 
    print('creating plots of results')
    
    modelPlot_path = results_path + '/modelPlot_results'
    chainPlot_path = results_path + '/chainPlot_results'
    cornerPlot_path = results_path + '/cornerPlot_results'
    chainList_path = results_path + '/chain_lists'
    
    red_X_squared = make_modelPlots(multi_band_list,kwargs_model,kwargs_result,
                                    kwargs_data,kwargs_psf, lens_info,
                                    lens_model_list,source_model_list,lens_light_model_list,
                                    mask_list,band_list,modelPlot_path,it+1,data_pairs_dicts[it]['object_ID'])
    
    printMemory('After modelPlot')
    
    save_chain_list(chain_list,chainList_path,it+1,data_pairs_dicts[it]['object_ID'])
    
    printMemory('After saving chain_list')
    
    del chain_list
    
    printMemory('After clearing chain_list')
    
    
#     make_chainPlots(chain_list, chainPlot_path, it+1, data_pairs_dicts[it]['object_ID'])
#     printMemory('After chainPlot')
    
#     make_cornerPlots(chain_list,cornerPlot_path,it+1, data_pairs_dicts[it]['object_ID'])
    
#     printMemory('After cornerPlot')
    
#     exec(open('Lens_Modeling_Auto/plot_results.py').read())
    
#     printMemory('After plot_results')
    
    csv_path = results_path
    
    
    #Create csv files
#     if it == 0:
    if not exists(csv_path + '/lens_results.csv'):
        exec(open('Lens_Modeling_Auto/create_csv.py').read())
#         exec(open('Lens_Modeling_Auto/create_csv_old.py').read())
    
    #Save results in csv file
    print('\n')
    print('writing model parameter results to csv files')
    
    print(kwargs_result)
    
    exec(open('Lens_Modeling_Auto/save_to_csv_full.py').read())
#     exec(open('Lens_Modeling_Auto/save_to_csv_full_old.py').read())
    
    ################################################################################################################# 
    
    ################################################ Model Shapelets ################################################
    
    #################################################################################################################
    
    if ((red_X_squared >= 1.5) and (use_shapelets == True)):
        n_max = 10
        print('\n')
        print('Reduced Chi^2 is still too high! I will now try modeling the source with shapelets with n_max = {}'.format(n_max))
        print('\n')
        
        source_model_list = ['SHAPELETS']
        
        multi_source_model_list = []
        
        for i in range(len(kwargs_data)):
            multi_source_model_list.extend(deepcopy(source_model_list))
        
        
        fixed_source = []
        kwargs_source_init = []
        kwargs_source_sigma = []
        kwargs_lower_source = []
        kwargs_upper_source = []
        
        beta_init = kwargs_result['kwargs_source'][0]['R_sersic'] / 3.
        #beta_init = 0.05
        
        fixed_source.append({'n_max': n_max,
                             'center_x': kwargs_result['kwargs_source'][0]['center_x'],
                             'center_y': kwargs_result['kwargs_source'][0]['center_y']})
        
        kwargs_source_init.append({'center_x': 0.01, 'center_y': 0.01, 'beta': beta_init})
        kwargs_source_sigma.append({'center_x': 0.01, 'center_y': 0.01, 'beta': 0.05})
        kwargs_lower_source.append({'center_x': -1.5, 'center_y': -1.5, 'beta': beta_init / np.sqrt(n_max + 1)})
        kwargs_upper_source.append({'center_x': 1.5, 'center_y': 1.5, 'beta': beta_init * np.sqrt(n_max + 1)})
        
        source_params_update = [[],[],[],[],[]]
        for i in range(len(kwargs_data)):
            source_params_update[0].extend(deepcopy(kwargs_source_init))
            source_params_update[1].extend(deepcopy(kwargs_source_sigma))
            source_params_update[2].extend(deepcopy(fixed_source))
            source_params_update[3].extend(deepcopy(kwargs_lower_source))
            source_params_update[4].extend(deepcopy(kwargs_upper_source))
            
        
        
        lens_params_update = deepcopy(lens_params)        
        lens_light_params_update = deepcopy(lens_light_params)

        lens_params_update[0] = deepcopy(kwargs_result['kwargs_lens'])
        #source_params_update[0] = deepcopy(kwargs_result['kwargs_source'])
        lens_light_params_update[0] = deepcopy(kwargs_result['kwargs_lens_light'])
        
        
        file = open(results_path+"/initial_params.txt","a")#append mode 
        file.write('\n')
        file.write('Addition of Shapelets: \n')
        file.write('\n')
        file.write("Model lists: \n")
        file.write("lens model: " + str(lens_model_list) + " \n")
        file.write("source model: " + str(multi_source_model_list) + " \n")
        file.write("lens light model: "+ str(multi_lens_light_model_list) + " \n")
        
        file.write("\n")
        file.write("kwargs_source (init,sigma,fixed,lower,upper): \n") 
#         file.write("\n")

        for i in range(len(source_params_update)):
#             file.write("\n")
            print(source_params_update[i], file=file)
#             file.write("\n")
        
        file.close()
        
#         SHAPELETS_indices = [i for i,x in enumerate(multi_source_model_list) if x == 'SHAPELETS']

#         for j in SHAPELETS_indices:
#              source_params_update[0][j]['beta'] = kwargs_result['kwargs_source'][j-1]['R_sersic']

        kwargs_params = {'lens_model': lens_params_update,
                        'source_model': source_params_update,
                        'lens_light_model': lens_light_params_update}
        
        kwargs_fixed = {'kwargs_lens': deepcopy(lens_params_update[2]), 
                 'kwargs_source': deepcopy(source_params_update[2]), 
                 'kwargs_lens_light': deepcopy(lens_light_params_update[2])}
        
        model_kwarg_names = get_kwarg_names(lens_model_list,multi_source_model_list,
                                         multi_lens_light_model_list,kwargs_fixed)
        #exec(open('Lens_Modeling_Auto/update_source_params_lists.py').read())
        
        
#         model_kwarg_names = get_kwarg_names(lens_model_list,multi_source_model_list,
#                                         multi_lens_light_model_list,kwargs_fixed)
       
        
        if this_is_a_test:
            fitting_kwargs_list = [['PSO', {'sigma_scale': 1, 'n_particles': 50, 'n_iterations': 100,'threadCount': numCores}]
                                    ,['MCMC', {'n_burn': 0, 'n_run': 10, 'walkerRatio': 10, 'sigma_scale': .05,'threadCount':numCores}]
                                  ]
        else:
            fitting_kwargs_list = [['PSO', {'sigma_scale': 1, 'n_particles': 200, 'n_iterations': 2000,'threadCount': numCores}]
                                    ,['MCMC', {'n_burn': 200, 'n_run': 800, 'walkerRatio': 10, 'sigma_scale': .05,'threadCount':numCores}]
                                  ]
        
        
        
#         fitting_kwargs_list = [['PSO', {'sigma_scale': 1, 'n_particles': 200, 'n_iterations': 2000,'threadCount': numCores}]
#                     ,['MCMC', {'n_burn': 200, 'n_run': 800, 'walkerRatio': 10, 'sigma_scale': .05}]]
#         fitting_kwargs_list = [['PSO', {'sigma_scale': 0.5, 'n_particles': 50, 'n_iterations': 100,'threadCount':numCores}],
#                               ['MCMC', {'n_burn': 0, 'n_run': 10, 'walkerRatio': 10, 'sigma_scale': .1,'threadCount':numCores}]]
    
        
        exec(open('Lens_Modeling_Auto/model_shapelets.py').read())
        
        toc3 = time.perf_counter()
        print('\n')
        print('Full sampling with shapelets (n_max = {}) took: {:.2f} minutes'.format(n_max,(toc3 - toc2)/60.0), '\n',
             'Total time: {:.2f} minutes'.format((toc3 - tic)/60.0))

        
    csv_path = results_path
    #Save results in csv file
    print('\n')
    print('writing model parameter results to csv files')
    
    print(kwargs_result)
    
    exec(open('Lens_Modeling_Auto/save_to_csv_lens.py').read())
#     exec(open('Lens_Modeling_Auto/save_to_csv_lens_old.py').read())
    print('\n')
    print('image {} modeling completed!'.format(it+1))
    
    printMemory('After save to csv/end of image')
    

    
    
    
