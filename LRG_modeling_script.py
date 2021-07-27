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
from Lens_Modeling_Auto.auto_modeling_functions import LRG_masking
from Lens_Modeling_Auto.fit_sequence_functions import initial_model_params
from Lens_Modeling_Auto.fit_sequence_functions import initial_modeling_fit
from Lens_Modeling_Auto.fit_sequence_functions import initial_fits_arcs_masked
from Lens_Modeling_Auto.fit_sequence_functions import initial_fits_arcs_masked_alt
from Lens_Modeling_Auto.fit_sequence_functions import full_sampling
from Lens_Modeling_Auto.plot_functions import make_modelPlots
from Lens_Modeling_Auto.plot_functions import make_chainPlots
from Lens_Modeling_Auto.plot_functions import make_cornerPlots
from Lens_Modeling_Auto.plot_functions import save_chain_list
from Lens_Modeling_Auto.plot_functions import plot_LRG_fit



#####################################################################################################################

#################################################### User Inputs ####################################################

#####################################################################################################################

# nohup python -u ./Lens_Modeling_Auto/LRG_modeling_script.py > LRG_modeling/results_mask/output_logs/output0_200.log &
      

# file paths to image data and results destination [TO DO BY USER]
data_path = '/LRG_data'
results_path = 'LRG_data/results_test'

if not exists(results_path):
    os.mkdir(results_path)

#Folder names for data, psf, noise map, original image [TO DO BY USER]
im_path = data_path + '/IMA' #add name of folder with image data
# im_path = data_path + '/simulations'
psf_path = data_path + '/PSF' #add name of folder with psf data
noise_path = data_path + '/RMS' #add name of folder with rms data, OR folder with FITS files that contain exposure times in header files (if using 'EXPTIME' for noise_type)
noise_type = 'NOISE_MAP' # 'NOISE_MAP' or 'EXPTIME'
band_list = ['r'] #list of bands
obj_name_location = 1 # index corresponding to which string of numbers in filenames are the ID 

#Modeling Options [TO DO BY USER]
use_shapelets = False #If True,then at the end of the modeling it tries shapelets instead of Sersic for the source profile if chi^2 is too large
fix_seed = False #bool. If True, uses saved seed values for each image from a previous modeling run
source_seed_path = '<previous results folder>/random_seed_init/' #path to seed values to be used
use_mask = True #bool; whether or not masks should be used in the modeling
mask_pickle_path = '<previous results folder>/masks/'#path to masks created previously. If None, new masks will be created by the script
Mask_rad_file = None #path to csv file or 'None'

#model lists [TO DO BY USER]
lens_model_list = []
source_model_list = []
lens_light_model_list = ['SERSIC_ELLIPSE']
this_is_a_test = False
mask_arcs = False
numCores = 1 


select_objects = None #Object Ids or None


# Additional info for images [TO DO BY USER]
deltaPix = 0.1857
zeroPt = 30
psf_upsample_factor = 2
ra_dec = None # 'csv', 'header', or 'None'
ra_dec_loc = None #path to csv file or header file, or 'None'
Mask_rad_file = None #'<path>.csv' #path to csv file or 'None'

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
        for j in range(len(df_info.loc[:,:])):
            if str(df_info.loc[j,'id']) == obj: RA,DEC = df_info.loc[j,'ra'],df_info.loc[j,'dec']
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
no_psf = []
no_noise_map = []
for i,x in enumerate(data_pairs_dicts): 
    if (not x['psf']): 
        print('******* MISSING PSF FILE!!! *******')
        print('ID: {}'.format(x['object_ID']))
        print('RA: {}, DEC: {}'.format(x['RA'],x['DEC']))
        print('\n')
        no_psf.append([x['object_ID'],x['RA'],x['DEC']])
        if (not x['noise_map']):
            no_noise_map.append([x['object_ID'],x['RA'],x['DEC']])
        continue
    elif (not x['noise_map']):
        print('******* MISSING Noise Map FILE!!! *******')
        print('ID: {}'.format(x['object_ID']))
        print('RA: {}, DEC: {}'.format(x['RA'],x['DEC']))
        no_noise_map.append([x['object_ID'],x['RA'],x['DEC']])
        print('\n')
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
    f.write('missing psf: {}'.format(no_psf))
    f.write('\n')
    f.write('missing noise map: {}'.format(no_noise_map))
    f.close()

printMemory('Before loop')

tic0 = time.perf_counter()

if not exists(results_path + "/Modeling_times.txt"):
    f = open(results_path + "/Modeling_times.txt","w")
    f.write('\n' + '###############################################################################################' + ' \n')
    f.write('\n')
    f.write('\n' + '######################################## Modeling Times #######################################' + ' \n')
    f.write('\n')
    f.write('\n' + '###############################################################################################' + ' \n')
    f.close()
    
for it in range(len(data_pairs_dicts[800:])):    
    it += 800 
    
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
    
    skip = False
    for f,x in enumerate(data_dict['image_data']):
        if (np.shape(x)[0] != np.shape(x)[1]) | (np.shape(data_dict['psf'][f])[0] != np.shape(data_dict['psf'][f])[1]) | (np.shape(data_dict['noise_map'][f])[0] != np.shape(data_dict['noise_map'][f])[1]):
            skip = True
    if skip:
        print('\n')
        print('Skipped because of image shape!!!')
        continue
        
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
    
   
            
#     if Mask_rad_file == None:
#         mask_size_ratio = None
#         mask_size_px,mask_size_as = estimate_radius(kwargs_data[0]['image_data'],
#                                                     deltaPix,center_x=c_x,center_y=c_y,show_plot=False, name = None)
#     else:
#         df_mask = pd.read_csv(Mask_rad_file)
#         mask_size_ratio = None
#         mask_size_as = float(df_mask.loc[df_mask[id_col_name] == int(data_pairs_dicts[it]['object_ID']),'dst_arcsec']) #+8.*deltaPix 
        
    
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
                    
#                 mask_gal = mask_for_sat(data['image_data'],deltaPix,
#                                               lens_rad_arcsec = gal_rad_as,
#                                               center_x=c_x,center_y=c_y,
#                                               lens_rad_ratio = None, 
#                                               show_plot = False)
#                 gal_mask_list.append(mask_gal)
                
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

                mask_size_px = 6.
                mask_size_as = mask_size_px * deltaPix
                mask,_ = LRG_masking(data['image_data'],deltaPix,c_x,c_y,show_plot = False,size = 6.,ax=None)

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
        
    else: 
        mask_list = None
        mask_dict_list = []
        for b in band_list:
            mask_dict = {}
            mask_dict['c_x'] = c_x
            mask_dict['c_y'] = c_y
            mask_dict['size arcsec'] = None
            mask_dict['size pixels'] = None
            mask_dict['mask'] = None
            mask_dict_list.append(mask_dict)
    
    if not mask_arcs:
        gal_mask_list = deepcopy(mask_list)
    
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
        fitting_kwargs_list = [['PSO', {'sigma_scale': 1, 'n_particles': 200, 'n_iterations': 2000,'threadCount': numCores}]
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
    
    if fix_seed:
        np.random.set_state(seed_val)
    else:
        np.random.seed(None)
        
    if not exists(save_seed_path):
        os.mkdir(save_seed_path)
    # get the initial state of the RNG
    save_seed_val = np.random.get_state()

    with open(save_seed_file, 'wb') as handle:
            pickle.dump(save_seed_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

   
    lens_initial_params,source_initial_params,lens_light_initial_params = initial_model_params(lens_model_list)
    
#     lens_light_initial_params[0]['center_x'],lens_light_initial_params[0]['center_y'] = c_x,c_y
    
    lens_light_params = [[],[],[],[],[]]
    source_params = [[],[],[],[],[]]
    lens_params = [[],[],[],[],[]]


    for j,f in enumerate(lens_light_params):
        for i in range(len(kwargs_data)):
            f.append(deepcopy(lens_light_initial_params[j]))

    
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
        fitting_kwargs_list = [['PSO', {'sigma_scale': 1, 'n_particles': 300, 'n_iterations': 2000,'threadCount': numCores}]
                                ,['MCMC', {'n_burn': 200, 'n_run': 2000, 'walkerRatio': 10, 'sigma_scale': .05,'threadCount':numCores}]
                              ]
    
    
    
    (chain_list, kwargs_result,kwargs_params,
     kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list, 
     kwargs_constraints) = full_sampling(fitting_kwargs_list,kwargs_params, 
                                                        kwargs_data, kwargs_psf,lens_model_list, 
                                                        source_model_list,lens_light_model_list,
                                                        mask_list)
    
#     if not this_is_a_test:
#         exec(open('Lens_Modeling_Auto/Full_Sampling.py').read())
    
    printMemory('After Full Sampling')

    toc1 = time.perf_counter()
    print('\n')
    print('Total time: {:.2f} minutes'.format((toc1 - tic)/60.0))
    
    f = open(results_path + "/Modeling_times.txt","a")
    f.write('\n')
    f.write('Main Sampling time: {:.4f} minutes'.format((toc1 - tic)/60.0))
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
    
    LRG_path = results_path 
    LRG_plot_path = LRG_path + '/Plot_kwargs/'
    if not exists(LRG_plot_path):
        os.mkdir(LRG_plot_path)
    
    
    LRG_plot_kwargs = {'multi_band_list': multi_band_list, 'kwargs_model': kwargs_model,
                   'kwargs_params': kwargs_result, 'likelihood_mask_list': mask_list}
    
    
    LRG_plot_kwargs,red_X_squared = plot_LRG_fit(LRG_plot_kwargs,band_list,modelPlot_path,it+1, data_pairs_dicts[it]['object_ID'])
    
    with open(LRG_plot_path + 'Image_{}-{}_plot_kwargs.pickle'.format(it+1,data_pairs_dicts[it]['object_ID']),'wb') as handle:
        pickle.dump(LRG_plot_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#     red_X_squared = make_modelPlots(multi_band_list,kwargs_model,kwargs_result,
#                                     kwargs_data,kwargs_psf, lens_info,
#                                     lens_model_list,source_model_list,lens_light_model_list,
#                                     mask_list,band_list,modelPlot_path,it+1,data_pairs_dicts[it]['object_ID'])
    
    printMemory('After modelPlot')
    
    save_chain_list(chain_list,chainList_path,it+1,data_pairs_dicts[it]['object_ID'])
    
    printMemory('After saving chain_list')
    
    del chain_list
    
    printMemory('After clearing chain_list')
    
    
    toc_end = time.perf_counter()
    image_model_time = (toc_end - tic)/60.0
    print('Modeling time for this image: {} minutes'.format((toc_end - tic)/60.0), '\n',
         'Total time of this modeling run: {} hours'.format((toc_end - tic0)/3600.0))
    
    print('\n')
    csv_path = results_path
    
    
    #     Create csv files
#     if it == 0:
    if not exists(csv_path + '/full_results.csv'):
        exec(open('Lens_Modeling_Auto/create_csv.py').read())
#         exec(open('Lens_Modeling_Auto/create_csv_old.py').read())
    
    #Save results in csv file
    print('\n')
    print('writing model parameter results to csv files')
    
    
    exec(open('Lens_Modeling_Auto/save_to_csv_full.py').read())
#     exec(open('Lens_Modeling_Auto/save_to_csv_full_old.py').read())
    
    
    print(kwargs_result)
    
    
    print('\n')
    print('image {} modeling completed!'.format(it+1))
    print('\n')
  
    
    f = open(results_path + "/Modeling_times.txt","a")
    f.write('\n')
    f.write('Modeling time for this image: {:.4f} minutes'.format((toc_end - tic)/60.0))
    f.write('\n')
    f.write('Total time of this modeling run: {:.4f} hours'.format((toc_end - tic0)/3600.0))
    f.write('\n')
    f.close()
    
    printMemory('After save to csv/end of image')
    

    
    
    
