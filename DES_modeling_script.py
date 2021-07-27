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
from Lens_Modeling_Auto.fit_sequence_functions import initial_fits_arcs_masked

from Lens_Modeling_Auto.fit_sequence_functions import full_sampling
from Lens_Modeling_Auto.plot_functions import make_modelPlots
from Lens_Modeling_Auto.plot_functions import make_chainPlots
from Lens_Modeling_Auto.plot_functions import make_cornerPlots
from Lens_Modeling_Auto.plot_functions import save_chain_list




#####################################################################################################################

#################################################### User Inputs ####################################################

#####################################################################################################################

# nohup python -u ./Lens_Modeling_Auto/DES_modeling_script.py > /results/output.log &

# print('lenstronomy version: {}'.format(lenstronomy.__version__))

# file paths to image data and results destination [TO DO BY USER]
data_path = 'DES_lenses' #path to image data
results_path = 'DES_lenses/results_test' #path to designated results folder

if not exists(results_path): #creates results folder if it doesn't already exist
    os.mkdir(results_path)

#Folder names for data, psf, noise map, original image [TO DO BY USER]
im_path = data_path + '/data'#add name of folder with image data
# im_path = data_path + '/simulations'
psf_path = data_path + '/psf' #add name of folder with psf data
noise_path = data_path + '/psf' #add name of folder with rms data, OR folder with FITS files that contain exposure times in header files (if using 'EXPTIME' for noise_type)
noise_type = 'EXPTIME' # 'NOISE_MAP' or 'EXPTIME'
band_list = ['g','r','i'] #list of bands
obj_name_location = 0 # index corresponding to which string of numbers in filenames are the ID 

#Modeling Options [TO DO BY USER]
use_shapelets = False #If True,then at the end of the modeling it tries shapelets instead of Sersic for the source profile if chi^2 is too large
fix_seed = False #bool. If True, uses saved seed values for each image from a previous modeling run
source_seed_path = '<previous results folder>/random_seed_init/' #path to seed values to be used
use_mask = True #bool; whether or not masks should be used in the modeling
mask_pickle_path = '<previous results folder>/masks/'#path to masks created previously. If None, new masks will be created by the script
Mask_rad_file = None #path to csv file or 'None'

#model lists
lens_model_list = ['SIE','SHEAR'] 
source_model_list = ['SERSIC_ELLIPSE']
lens_light_model_list = ['SERSIC_ELLIPSE']
point_source_model_list = None
this_is_a_test = False #If true, changes PSO and MCMC settings to make modeling very fast (for debugging/troubleshooting)
numCores = 1 # number of CPUs to use 

#path to Reff and n_s source distributions that lenstronomy uses for kde prior method. 
#Warning: Method is very slow. Better to set to None
kde_prior_path = None #'/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/kde_priors/'
if kde_prior_path != None:
    with open(kde_prior_path + 'R_source.pickle', 'rb') as handle:
        kde_Rsource = pickle.load(handle)
        
    with open(kde_prior_path + 'n_source.pickle', 'rb') as handle:
        kde_nsource = pickle.load(handle)
else:
    kde_Rsource = None
    kde_nsource = None

#specify IDs of specific images to model. Otherwise model all images in data folder 
select_objects =  None #['03310601','06653211','06788344','14083401',
#                     '14327423','15977522','16033319','17103670',
#                     '19990514']  #List of strings with object IDs, or None

# Additional info for images [TO DO BY USER]
deltaPix = 0.27 #pixel scale of the images in arcsec/pixel
zeroPt = 30 #not used anywhere
psf_upsample_factor = 1 #If psf is upsampled
ra_dec = 'csv' # 'csv', 'header', or 'None'. Where to find ra and dec values if desired for naming. Otherwise will have 'N/A' in RA and DEC columns of results
ra_dec_loc = '<path>.csv' #path to csv file or header file, or 'None'
id_col_name = 'id_1' #column in csv file to look for image IDs

printMemory('Beginning')

#####################################################################################################################

########################################### Find Data and sort filenames ############################################

#####################################################################################################################


#find files
im_files = [f for f in listdir(im_path) if isfile('/'.join([im_path,f]))]
psf_files,noise_files = [],[]
psf_files_dict, noise_files_dict = {},{}

for b in band_list: 
    psf_files.append([f for f in listdir(psf_path + '/' + b) if isfile('/'.join([psf_path + '/' + b,f]))])
    noise_files.append([f for f in listdir(noise_path + '/' + b) if isfile('/'.join([noise_path + '/' + b,f]))])
    psf_files_dict[b] = [f for f in listdir(psf_path + '/' + b) if isfile('/'.join([psf_path + '/' + b,f]))]
    noise_files_dict[b] = [f for f in listdir(noise_path + '/' + b) if isfile('/'.join([noise_path + '/' + b,f]))]



#Extract object IDs from filenames     
obj_names = []
if not select_objects:
    for x in im_files:
        obj_names.append(re.findall('\d+', x)[obj_name_location])
else: obj_names = deepcopy(select_objects)

#sort all file names and info into list of dicts
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
            if float(df_info.loc[j,id_col_name]) == float(obj): RA,DEC = df_info.loc[j,'ra'],df_info.loc[j,'dec']
    else: RA, DEC = 'N/A','N/A'
    
    data_pairs_dicts.append({'image_data': im , 'psf': psf , 'noise_map': noise, 
                             'noise_type': noise_type,'object_ID': str(int(obj)),'RA': RA, 'DEC': DEC})

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
if not exists(results_path + "/initial_params.txt"):
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

tic0 = time.perf_counter() #start timer

if not exists(results_path + "/Modeling_times.txt"):
    f = open(results_path + "/Modeling_times.txt","w")
    f.write('\n' + '###############################################################################################' + ' \n')
    f.write('\n')
    f.write('\n' + '######################################## Modeling Times #######################################' + ' \n')
    f.write('\n')
    f.write('\n' + '###############################################################################################' + ' \n')
    f.close()
    
for it in range(len(data_pairs_dicts[:7])):    
#     it += 45 
    
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
    
    c_x,c_y = find_lens_gal(kwargs_data[-1]['image_data'],deltaPix,show_plot=False,title=data_pairs_dicts[it]['object_ID'])
    
    if Mask_rad_file == None:
        mask_size_ratio = None
        mask_size_px,mask_size_as = estimate_radius(kwargs_data[0]['image_data'],
                                                    deltaPix,center_x=c_x,center_y=c_y,show_plot=False, name = None)
    else:
        df_mask = pd.read_csv(Mask_rad_file)
        mask_size_ratio = None
        mask_size_as = float(df_mask.loc[df_mask[id_col_name] == int(data_pairs_dicts[it]['object_ID']),'dst_arcsec']) #+8.*deltaPix 
        
    
    gal_mask_list = []
    gal_rad_as = 5 * deltaPix  
    mask_list = []
    mask_dict_list = []
#     sizes_As = []
#     sizes_px = []

    if use_mask:
        if mask_pickle_path != None:
            print('Using saved mask instead of creating one')
#             mask_list = []
            for k,data in enumerate(kwargs_data): 
                with open(mask_pickle_path + '{}/{}.pickle'.format(band_list[k],data_pairs_dicts[it]['object_ID']), 'rb') as handle:
                    mask_dict = pickle.load(handle)
                    mask_list.append(mask_dict['mask'])
                    mask_dict_list.append(mask_dict)
                    
                mask_gal = mask_for_sat(data['image_data'],deltaPix,
                                              lens_rad_arcsec = gal_rad_as,
                                              center_x=c_x,center_y=c_y,
                                              lens_rad_ratio = None, 
                                              show_plot = False)
                gal_mask_list.append(mask_gal)
                
                mask_path = results_path + '/masks'
                
                if mask_pickle_path != mask_path:
                    if not exists(mask_path):
                        os.mkdir(mask_path)
                    band_path = mask_path + '/' + band_list[k]
                    
                    if not exists(band_path):
                        os.mkdir(band_path)
                    
                    with open(band_path + '/{}.pickle'.format(data_pairs_dicts[it]['object_ID']), 'wb') as handle:
                        pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    
            
        else:
            for k,data in enumerate(kwargs_data): 
                if not exists(results_path + '/masks'):
                    os.mkdir(results_path + '/masks')
                mask_path = results_path + '/masks'
                
                band_path = mask_path + '/' + band_list[k]
                if not exists(band_path):
                    os.mkdir(band_path)

                mask = mask_for_sat(data['image_data'],deltaPix,
                                              lens_rad_arcsec = mask_size_as,
                                              center_x=c_x,center_y=c_y,
                                              lens_rad_ratio = mask_size_ratio, 
                                              show_plot = False)
                mask_list.append(mask)


                mask_dict = {}
                mask_dict['c_x'] = c_x
                mask_dict['c_y'] = c_y
                mask_dict['size arcsec'] = mask_size_as
                mask_dict['size pixels'] = mask_size_px
                mask_dict['mask'] = mask
                mask_dict_list.append(mask_dict)
                with open(band_path + '/{}.pickle'.format(data_pairs_dicts[it]['object_ID']), 'wb') as handle:
                        pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #             sizes_As.append(size_Arcsec)
        #             sizes_px.append(size_pix)
        
    else: mask_list = None
        
#     if not mask_arcs:
#         gal_mask_list = deepcopy(mask_list) 
    
    
    file = open(results_path+"/initial_params.txt","a")#append mode 
    file.write("Mask Size: \n")
    file.write("{} pixels,{} arcsec \n".format(mask_dict_list[0]['size pixels'],mask_dict_list[0]['size arcsec']))
    file.write("Mask Center: \n")
    file.write("({},{}) \n".format(mask_dict_list[0]['c_x'],mask_dict_list[0]['c_y']))
    if mask_pickle_path != None:
        file.write(mask_pickle_path)
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

    if fix_seed:
        with open(source_seed_path + '{}.pickle'.format(data_pairs_dicts[it]['object_ID']), 'rb') as handle:
            seed_val = pickle.load(handle)
            print('Using seed from: {}'.format(source_seed_path))
            print(seed_val)
    else: seed_val = None
    
    name = '{}.pickle'.format(data_pairs_dicts[it]['object_ID'])
    save_seed_path = results_path + '/random_seed_init/'
    save_seed_file = save_seed_path + name
    
    init_chainList_path = results_path + '/chain_lists_init/'
    init_chainList_file = init_chainList_path + name
    if not exists(save_seed_path):
        os.mkdir(save_seed_path)
    if not exists(init_chainList_path):
        os.mkdir(init_chainList_path)
    
    
    (lens_initial_params,
     source_initial_params,
     lens_light_initial_params,
     ps_initial_params) = initial_model_params(lens_model_list,point_source_model_list = point_source_model_list)
    
#     (kwargs_params,kwargs_fixed, kwargs_result,
#      chain_list,kwargs_likelihood, kwargs_model, 
#      kwargs_data_joint, multi_band_list, 
#      kwargs_constraints) = initial_modeling_fit(fitting_kwargs_list,lens_model_list,source_model_list,
#                                                 lens_light_model_list,lens_initial_params,
#                                                 source_initial_params,lens_light_initial_params,
#                                                 kwargs_data,kwargs_psf,mask_list,fix_seed = fix_seed,
#                                                 fix_seed_val = seed_val,save_seed_file = save_seed_file, 
#                                                 chainList_file = init_chainList_file)
    
    (kwargs_params,kwargs_fixed, kwargs_result,
     chain_list,kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list, 
     kwargs_constraints)= initial_fits_arcs_masked(fitting_kwargs_list,lens_model_list,
                                                 source_model_list,lens_light_model_list,
                                                 lens_initial_params,source_initial_params,
                                                 lens_light_initial_params,
                                                 kwargs_data,
                                                 kwargs_psf,mask_list = mask_list, 
                                                 gal_mask_list = gal_mask_list,
                                                 kde_nsource=kde_nsource,kde_Rsource=kde_Rsource,
                                                 fix_seed = fix_seed,fix_seed_val = seed_val,
                                                 save_seed_file = save_seed_file, 
                                                 chainList_file = init_chainList_file,
                                                 ps_model_list =point_source_model_list,
                                                 ps_initial_params = ps_initial_params)

#     exec(open('Lens_Modeling_Auto/initial_modeling_fit.py').read())
    
    printMemory('After initial fit')
    
    
    toc1 = time.perf_counter()                
    
    print('\n')
    print('First sampling took: {:.2f} minutes'.format((toc1 - tic)/60.0))
    
    f = open(results_path + "/Modeling_times.txt","a")
    f.write('\n')
    f.write('Image: {}'.format(it+1))
    f.write('\n')
    f.write('Pre-sampling optimization time: {:.4f} minutes'.format((toc1 - tic)/60.0))
    f.close()
    
    
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
    
    
    
    (chain_list, kwargs_result,kwargs_params,
     kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list, 
     kwargs_constraints) = full_sampling(fitting_kwargs_list,kwargs_params, 
                                                        kwargs_data, kwargs_psf,lens_model_list, 
                                                        source_model_list,lens_light_model_list,
                                                        kde_nsource=kde_nsource,
                                                        kde_Rsource=kde_Rsource,
                                                        mask_list=mask_list,
                                                        ps_model_list = point_source_model_list)
    
#     if not this_is_a_test:
#         exec(open('Lens_Modeling_Auto/Full_Sampling.py').read())
    
    printMemory('After Full Sampling')

    toc2 = time.perf_counter()
    print('\n')
    print('Full sampling took: {:.2f} minutes'.format((toc2 - toc1)/60.0), '\n',
         'Total time: {:.2f} minutes'.format((toc2 - tic)/60.0))
    
    f = open(results_path + "/Modeling_times.txt","a")
    f.write('\n')
    f.write('Main Sampling time: {:.4f} minutes'.format((toc2 - toc1)/60.0))
    f.close()
    
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
    
    toc3 = time.perf_counter()
    image_model_time = (toc3 - tic)/60.0
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
    print('\n')
    toc_end = time.perf_counter()
    print('Modeling time for this image: {} minutes'.format((toc_end - tic)/60.0), '\n',
         'Total time of this modeling run: {} hours'.format((toc_end - tic0)/3600.0))
    
    print('\n')
    
    f = open(results_path + "/Modeling_times.txt","a")
    f.write('\n')
    f.write('Modeling time for this image: {:.4f} minutes'.format((toc_end - tic)/60.0))
    f.write('\n')
    f.write('Total time of this modeling run: {:.4f} hours'.format((toc_end - tic0)/3600.0))
    f.write('\n')
    f.close()
    
    printMemory('After save to csv/end of image')
    

    
    
    
