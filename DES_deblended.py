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
from Lens_Modeling_Auto.fit_sequence_functions import initial_fits_arcs_masked_alt
from Lens_Modeling_Auto.fit_sequence_functions import model_deblended
from Lens_Modeling_Auto.fit_sequence_functions import model_deblended_long
from Lens_Modeling_Auto.fit_sequence_functions import full_sampling
from Lens_Modeling_Auto.plot_functions import make_modelPlots
from Lens_Modeling_Auto.plot_functions import make_chainPlots
from Lens_Modeling_Auto.plot_functions import make_cornerPlots
from Lens_Modeling_Auto.plot_functions import save_chain_list
from lenstronomy.Util.mask_util import mask_center_2d




#####################################################################################################################

#################################################### User Inputs ####################################################

#####################################################################################################################

# nohup python -u ./Lens_Modeling_Auto/DES_deblended.py > lens_candidates/Group1/muscadet_deblended/results_short/output_logs/output0_8.log &
      

# file paths to image data and results destination [TO DO BY USER]
# data_path = '/home/astro/maus/Desktop/LASTRO_lab/Specialization_Project/ringcatalog'
# results_path = '/home/astro/maus/Desktop/LASTRO_lab/Specialization_Project/ringcatalog/results_full_catalog'
data_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1'
results_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/muscadet_deblended/results_short'

if not exists(results_path):
    os.mkdir(results_path)

#Folder names for data, psf, noise map, original image [TO DO BY USER]
im_path = data_path + '/data'
deblended_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/MuSCADeT_models_v4.pkl'
deblended_path_alt = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/MuSCADeT_models_v4_2.pkl'
# LRG_path = data_path + 'output_of_the_network_rescaled/LRG'
# source_path = data_path + 'output_of_the_network_rescaled/sources'
# im_path = data_path + '/simulations'
psf_path = data_path + '/psf'
noise_path = data_path + '/psf'
noise_type = 'EXPTIME'
band_list = ['g','r','i']
obj_name_location = 0

kde_prior_path = None #'/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/kde_priors/'
if kde_prior_path != None:
    with open(kde_prior_path + 'R_source.pickle', 'rb') as handle:
        kde_Rsource = pickle.load(handle)
        
    with open(kde_prior_path + 'n_source.pickle', 'rb') as handle:
        kde_nsource = pickle.load(handle)
else:
    kde_Rsource = None
    kde_nsource = None

#Modeling Options [TO DO BY USER]
#Modeling Options [TO DO BY USER]
fix_seed = False
source_seed_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/SIE_lens/results_Ap30/random_seed_init_new/'
use_shapelets = False
use_mask = True
mask_center = True
# mask_pickle_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/muscadet_deblended/custom_masks/masks/'
mask_pickle_path ='/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/muscadet_deblended/results_Jun24/masks/'

lens_model_list = ['SIE','SHEAR']
source_model_list = ['SERSIC_ELLIPSE']
lens_light_model_list = ['SERSIC_ELLIPSE']
point_source_model_list = None
this_is_a_test = False
numCores = 1 

select_objects =  None
# ['03310601','06653211','06788344','14083401',
#                      '14327423','15977522','16033319','17103670',
#                     '19990514']


# Additional info for images [TO DO BY USER]
deltaPix = 0.27
zeroPt = 30
psf_upsample_factor = 1
ra_dec = 'csv' # 'csv', 'header', or 'None'
ra_dec_loc = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/group1_v2.csv'#path to csv file or header file, or 'None'
Mask_rad_file = None #'/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/mask_v2.csv' #path to csv file or 'None'

id_col_name = 'id_1'

printMemory('Beginning')

#####################################################################################################################

########################################### Find Data and sort filenames ############################################

#####################################################################################################################

#unpack deblended pkl file:
with open(deblended_path, 'rb') as handle:
    data_structure = pickle.load(handle)
    
with open(deblended_path_alt, 'rb') as handle:
    data_structure_alt = pickle.load(handle)
    
LRG_all_data = deepcopy(data_structure[2])
source_all_data = deepcopy(data_structure[3])

LRG_all_data_alt = deepcopy(data_structure_alt[2])
source_all_data_alt = deepcopy(data_structure_alt[3])

im_files = [f for f in listdir(im_path) if isfile('/'.join([im_path,f]))]
# LRG_files = [f for f in listdir(LRG_path) if isfile('/'.join([LRG_path,f]))]
# source_files = [f for f in listdir(source_path) if isfile('/'.join([source_path,f]))]

# im_files = deepcopy(data_structure[0])

psf_files,noise_files = [],[]
psf_files_dict, noise_files_dict = {},{}

for b in band_list: 
    psf_files.append([f for f in listdir(psf_path + '/' + b) if isfile('/'.join([psf_path + '/' + b,f]))])
    noise_files.append([f for f in listdir(noise_path + '/' + b) if isfile('/'.join([noise_path + '/' + b,f]))])
    psf_files_dict[b] = [f for f in listdir(psf_path + '/' + b) if isfile('/'.join([psf_path + '/' + b,f]))]
    noise_files_dict[b] = [f for f in listdir(noise_path + '/' + b) if isfile('/'.join([noise_path + '/' + b,f]))]


# print(im_files[:10])
    
obj_names = []
if not select_objects:
    for x in im_files:
        obj_names.append(re.findall('\d+', x)[obj_name_location])
else: obj_names = deepcopy(select_objects)

# obj_names = obj_names[:10]

data_pairs_dicts = []
for i,obj in enumerate(obj_names):
    for x in im_files:
        if int(obj) == int(re.findall('\d+', x)[obj_name_location]): im = x
            
#     for y in LRG_files:
#         if obj == re.findall('\d+', y)[obj_name_location]: LRG = y
            
#     for z in source_files:
#         if obj == re.findall('\d+', z)[obj_name_location]: source = z

    psf = {}
    for b in band_list:
        for file in psf_files_dict[b]:
            if int(obj) == int(re.findall('\d+', file)[obj_name_location]): psf[b] = '/'.join([b,file])

    noise = {}
    for b in band_list:
        for file in noise_files_dict[b]:
            if int(obj) == int(re.findall('\d+', file)[obj_name_location]): noise[b]= '/'.join([b,file])

    if ra_dec == 'csv':
        df_info = pd.read_csv(ra_dec_loc)
        for j in range(len(df_info.loc[:,:])):
            if int(df_info.loc[j,'id']) == int(obj): RA,DEC = df_info.loc[j,'ra'],df_info.loc[j,'dec']
    else: RA, DEC = 'N/A','N/A'
    
    data_pairs_dicts.append({'image_data': im , 
#                              'LRG_data': LRG,'source_data': source,
                             'psf': psf , 'noise_map': noise, 
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
#     if (not x['psf']) or (not x['noise_map']) or (not x['LRG_data']) or (not x['source_data']):
        continue
    count += 1
    print('image {}'.format(count))
    print('ID: {}'.format(x['object_ID']))
    print('RA: {}, DEC: {}'.format(x['RA'],x['DEC']))
    print('Full Image data: ',x['image_data'])
#     print('LRG data: ',x['LRG_data'])
#     print('Lensed source data: ',x['source_data'])
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

tic0 = time.perf_counter()

f = open(results_path + "/Modeling_times.txt","w")
f.write('\n' + '###############################################################################################' + ' \n')
f.write('\n')
f.write('\n' + '######################################## Modeling Times #######################################' + ' \n')
f.write('\n')
f.write('\n' + '###############################################################################################' + ' \n')
f.close()

for it in range(len(data_pairs_dicts[48:])):    
    it += 48
    
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
    
    file = data_pairs_dicts[it]['image_data']
    
    #band_index = np.where(np.array(band_list) == band)[0][0]
    data,hdr = openFITS(im_path + '/' + file)
#     LRG_data,_ = openFITS(LRG_path + '/' + data_pairs_dicts[it]['LRG_data'])
#     source_data,_ = openFITS(source_path + '/' + data_pairs_dicts[it]['source_data'])

    
    
    if file in data_structure[0]:
        index = data_structure[0].index(file)
        LRG_data = deepcopy(LRG_all_data[index])
        source_data = deepcopy(source_all_data[index])
    else:
        index = data_structure_alt[0].index(file)
        LRG_data = deepcopy(LRG_all_data_alt[index])
        source_data = deepcopy(source_all_data_alt[index])
        
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


    data_dict = {'image_data': [], 'LRG_data': [], 'source_data': [],'image_hdr': [], 
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
        
        if np.ndim(LRG_data) == 4:
            data_dict['LRG_data'].append(LRG_data[0][i])
        elif np.ndim(data) == 3:
            data_dict['LRG_data'].append(LRG_data[i])
            
        if np.ndim(source_data) == 4:
            data_dict['source_data'].append(source_data[0][i])
        elif np.ndim(data) == 3:
            data_dict['source_data'].append(source_data[i])
#         data_dict['image_hdr'].append(hdr[0])
    
    print('calculating background values')
    print('\n')
    background_rms = calcBackgroundRMS(data_dict['image_data']) #calculate rms background
    background_rms_LRG = calcBackgroundRMS(data_dict['LRG_data'])
    background_rms_source = calcBackgroundRMS(data_dict['source_data'])
    print('\n')

    lens_info = []
    LRG_info = []
    source_info = []

    for i,x in enumerate(data_dict['image_data']):

        lens_info.append({'deltaPix': deltaPix ,
                         'numPix': len(x),
                         'background_rms': background_rms[i],
                         'psf_type': 'PIXEL',
                         'psf_upsample_factor': psf_upsample_factor})

        if noise_type == 'EXPTIME': 
            lens_info[i]['exposure_time'] = data_dict['noise_hdr'][i][0]['EXPTIME']
#             lens_info[i]['exposure_time'] = 800.
            lens_info[i]['noise_map'] = None
        else:
            lens_info[i]['exposure_time'] = None
            lens_info[i]['noise_map'] = data_dict['noise_map'][i] 
            
    for i,x in enumerate(data_dict['LRG_data']):

        LRG_info.append({'deltaPix': deltaPix ,
                         'numPix': len(x),
                         'background_rms': background_rms_LRG[i],
                         'psf_type': 'PIXEL',
                         'psf_upsample_factor': psf_upsample_factor})

        if noise_type == 'EXPTIME': 
            LRG_info[i]['exposure_time'] = data_dict['noise_hdr'][i][0]['EXPTIME']
#             LRG_info[i]['exposure_time'] = 800.
            LRG_info[i]['noise_map'] = None
        else:
            LRG_info[i]['exposure_time'] = None
            LRG_info[i]['noise_map'] = data_dict['noise_map'][i] 
            
    for i,x in enumerate(data_dict['source_data']):

        source_info.append({'deltaPix': deltaPix ,
                         'numPix': len(x),
                         'background_rms': background_rms_source[i],
                         'psf_type': 'PIXEL',
                         'psf_upsample_factor': psf_upsample_factor})

        if noise_type == 'EXPTIME': 
            source_info[i]['exposure_time'] = data_dict['noise_hdr'][i][0]['EXPTIME']
#             source_info[i]['exposure_time'] = 800.
            source_info[i]['noise_map'] = None
        else:
            source_info[i]['exposure_time'] = None
            source_info[i]['noise_map'] = data_dict['noise_map'][i] 

    kwargs_data, kwargs_psf = prepareData(lens_info,data_dict['image_data'],
                                           data_dict['psf']) 
    kwargs_data_LRG, kwargs_psf = prepareData(LRG_info,data_dict['LRG_data'],
                                           data_dict['psf']) 
    kwargs_data_source, kwargs_psf = prepareData(source_info,data_dict['source_data'],
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
    gal_rad_as = 6 * deltaPix  
    mask_list = []
    mask_dict_list = []
    source_mask_list = []
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
        
      
        if mask_center:
            for i,m in enumerate(mask_dict_list):
                source_mask = deepcopy(m['mask'])
                numPix = len(source_mask)
                center_mask = np.zeros([numPix,numPix])

                for j in range(numPix):
                    center_mask[j] = mask_center_2d(c_x,c_y, 3, np.linspace(0,numPix - 1,numPix), j)
                source_mask[center_mask == 0] = 0
                source_mask_list.append(source_mask)
        else:
            source_mask_list = deepcopy(mask_list)
        
    else: mask_list = None
        
    
    
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
    
    
    (lens_initial_params,source_initial_params,
     lens_light_initial_params,ps_initial_params) = initial_model_params(lens_model_list)
    
    
#     (kwargs_params,kwargs_fixed, kwargs_result,
#      chain_list,kwargs_likelihood, kwargs_model, kwargs_data_joint, 
#      multi_band_list, kwargs_constraints) = model_deblended_long(fitting_kwargs_list,lens_model_list,
#                                                         source_model_list,lens_light_model_list,
#                                                         lens_initial_params,source_initial_params,
#                                                         lens_light_initial_params,kwargs_data,
#                                                         kwargs_data_LRG,kwargs_data_source,kwargs_psf,
#                                                         num = it+1,object_ID = data_pairs_dicts[it]['object_ID'],
#                                                         mask_list = mask_list,
#                                                         source_mask_list = source_mask_list,
#                                                         gal_mask_list = gal_mask_list,fix_seed = fix_seed,
#                                                         fix_seed_val = seed_val,save_seed_file = save_seed_file, 
#                                                         chainList_file = init_chainList_file,
#                                                         kde_nsource=kde_nsource,kde_Rsource=kde_Rsource,
#                                                         results_path = results_path, band_list = band_list
#                                                         )
    (kwargs_params,kwargs_fixed, kwargs_result,
     chain_list,kwargs_likelihood, kwargs_model, kwargs_data_joint, 
     multi_band_list, kwargs_constraints) = model_deblended(fitting_kwargs_list,lens_model_list,
                                                        source_model_list,lens_light_model_list,
                                                        lens_initial_params,source_initial_params,
                                                        lens_light_initial_params,kwargs_data,
                                                        kwargs_data_LRG,kwargs_data_source,kwargs_psf,
                                                        num = it+1,object_ID = data_pairs_dicts[it]['object_ID'],
                                                        mask_list = mask_list,
                                                        source_mask_list = source_mask_list,
                                                        gal_mask_list = gal_mask_list,fix_seed = fix_seed,
                                                        fix_seed_val = seed_val,save_seed_file = save_seed_file, 
                                                        chainList_file = init_chainList_file,
                                                        results_path = results_path, band_list = band_list
                                                        )
    
#     kwargs_params,kwargs_fixed, kwargs_result,
#     chain_list,kwargs_likelihood, kwargs_model, 
#     kwargs_data_joint, multi_band_list, kwargs_constraints = initial_modeling_fit(fitting_kwargs_list,lens_model_list,source_model_list,
#                                           lens_light_model_list,lens_initial_params,
#                                           source_initial_params,lens_light_initial_params,
#                                           kwargs_data,kwargs_psf,mask_list)
    
#     kwargs_params,kwargs_fixed, kwargs_result,chain_list,kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list, kwargs_constraints= initial_fits_arcs_masked(fitting_kwargs_list,lens_model_list,
#                                                  source_model_list,lens_light_model_list,
#                                                  lens_initial_params,source_initial_params,
#                                                  lens_light_initial_params,kwargs_data,
#                                                  kwargs_psf,mask_list = mask_list,
#                                                  gal_mask_list = gal_mask_list)
    
#     kwargs_params,kwargs_fixed, kwargs_result,chain_list,kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list, kwargs_constraints= initial_fits_arcs_masked_alt(fitting_kwargs_list,lens_model_list,
#                                                  source_model_list,lens_light_model_list,
#                                                  lens_initial_params,source_initial_params,
#                                                  lens_light_initial_params,kwargs_data,
#                                                  kwargs_psf,mask_list = mask_list,
#                                                  gal_mask_list = gal_mask_list)

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
    
    
    
    (chain_list,kwargs_result,kwargs_params,
     kwargs_likelihood, kwargs_model, 
     kwargs_data_joint, multi_band_list, 
     kwargs_constraints) = full_sampling(fitting_kwargs_list,kwargs_params,kwargs_data, 
                                         kwargs_psf,lens_model_list,source_model_list,
                                         lens_light_model_list,
                                         kde_nsource=kde_nsource,
                                         kde_Rsource=kde_Rsource,
                                         mask_list=mask_list)
    
#     if not this_is_a_test:
#         exec(open('Lens_Modeling_Auto/Full_Sampling.py').read())
    
    printMemory('After Full Sampling')

    toc2 = time.perf_counter()
    print('\n')
    print('Full sampling took: {:.2f} minutes'.format((toc2 - toc1)/60.0), '\n',
         'Total time for this image: {:.2f} minutes'.format((toc2 - tic)/60.0))
    
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
            fitting_kwargs_list = [['PSO', {'sigma_scale': 1, 'n_particles': 300, 'n_iterations': 2000,'threadCount': numCores}]
                                    ,['MCMC', {'n_burn': 200, 'n_run': 800, 'walkerRatio': 10, 'sigma_scale': .05,'threadCount':numCores}]
                                  ]
        
        
        
#         fitting_kwargs_list = [['PSO', {'sigma_scale': 1, 'n_particles': 200, 'n_iterations': 2000,'threadCount': numCores}]
#                     ,['MCMC', {'n_burn': 200, 'n_run': 800, 'walkerRatio': 10, 'sigma_scale': .05}]]
#         fitting_kwargs_list = [['PSO', {'sigma_scale': 0.5, 'n_particles': 50, 'n_iterations': 100,'threadCount':numCores}],
#                               ['MCMC', {'n_burn': 0, 'n_run': 10, 'walkerRatio': 10, 'sigma_scale': .1,'threadCount':numCores}]]
    
        
        
        
        
        
        exec(open('Lens_Modeling_Auto/model_shapelets.py').read())
        toc4 = time.perf_counter()
        shapelets_modeling_time = (toc4 - tic)/60.0
        print('\n')
        print('Full sampling with shapelets (n_max = {}) took: {:.2f} minutes'.format(n_max,(toc4 - toc3)/60.0), '\n',
             'Total time: {:.2f} minutes'.format((toc4 - tic)/60.0))

        
    csv_path = results_path
    #Save results in csv file
    print('\n')
    print('writing model parameter results to csv files')
    
    print(kwargs_result)
    
    toc_end = time.perf_counter()
    image_model_time = (toc_end - tic)/60.0
    
    exec(open('Lens_Modeling_Auto/save_to_csv_lens.py').read())
#     exec(open('Lens_Modeling_Auto/save_to_csv_lens_old.py').read())
    print('\n')
    print('image {} modeling completed!'.format(it+1))
    print('\n')
    
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
    

    
    
    
