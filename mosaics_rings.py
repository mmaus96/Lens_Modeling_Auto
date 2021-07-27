import sys
if sys.platform[:5] == 'linux':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os
from os import walk
from os import listdir
from os.path import isfile, join, isdir, exists
import time
import numpy as np
import pandas as pd
from Lens_Modeling_Auto.auto_modeling_functions_1p0 import openFITS
from Lens_Modeling_Auto.auto_modeling_functions_1p0 import find_components
from Lens_Modeling_Auto.auto_modeling_functions import find_lens_gal
from Lens_Modeling_Auto.auto_modeling_functions_1p0 import calcBackgroundRMS
from Lens_Modeling_Auto.auto_modeling_functions_1p0 import prepareData
from Lens_Modeling_Auto.auto_modeling_functions_1p0 import prepareFit
from Lens_Modeling_Auto.auto_modeling_functions_1p0 import mask_for_sat
from Lens_Modeling_Auto.auto_modeling_functions_1p0 import mask_for_lens_gal
from Lens_Modeling_Auto.auto_modeling_functions_1p0 import estimate_radius
from Lens_Modeling_Auto.auto_modeling_functions import df_2_kwargs_results
from Lens_Modeling_Auto.plot_functions import plot_data
from Lens_Modeling_Auto.plot_functions import subtract_from_data_plot
from Lens_Modeling_Auto.plot_functions import normalized_residual_plot
from Lens_Modeling_Auto.plot_functions import magnification_plot
from Lens_Modeling_Auto.plot_functions import convergence_plot
from Lens_Modeling_Auto.plot_functions import source_plot
from Lens_Modeling_Auto.plot_functions import plot_line_set
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.Plots import plot_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Plots import lens_plot

import matplotlib.colors as colors
from functools import reduce
from matplotlib.colors import SymLogNorm
import pickle
import re
from matplotlib.patches import Circle
from copy import deepcopy
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Analysis.image_reconstruction import ModelBand
from math import ceil


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        

path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/Rings_catalog/'
csv_path = path + 'new_priors/results_Jun11/'
masks_path = path + 'new_priors/results_Jun11/masks/'
results_path = path + 'new_priors/results_Jun11' 
everything_in_df = True
names_path = None#'/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/SL.csv'
ra_dec_path = path + 'Rings_cat_sv.csv'
id_col_name = 'id'
fixed_mask_path = None#'/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/SIE_lens/custom_masks/results_new_mask_final/'
df_fixed_mask = None#pd.read_csv(fixed_mask_path + 'full_results_sorted.csv',delimiter =',')
select_objects =  None
# ['3310601','6653211','6788344','14083401',
#                     '14327423','15977522','16033319','17103670',
#                     '19990514']


# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/ringcatalog/'
# csv_path = path + 'results_May3/'
# results_path = path + 'results_May3'

# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_lenses/'
# csv_path = path + 'Sure_Lens/SIE_lens/results_May31/'
# masks_path = path + 'Sure_Lens/SIE_lens/results_May31/masks'
# results_path = path + 'Sure_Lens/SIE_lens/results_May31'
# everything_in_df = True
# names_path = None
# ra_dec_path = path + 'lenses_coord.csv'
# name_root = 'CFIS'
# id_col_name = 'idPS1'
# fixed_mask_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_lenses/Sure_Lens/SIE_lens/test_best_seed/test_final/'
# df_fixed_mask = pd.read_csv(fixed_mask_path + 'full_results.csv',delimiter =',')
# # select_objects = ['144641749689622225','145851483996593559','146212542943478163',
# #                     '149131184425371844','149231702242056192']
# select_objects = ['144641749689622225','146212542943478163',
#                     '149131184425371844','149231702242056192']
# df_cut = df_fixed_mask.loc[df_fixed_mask['ID'].isin(np.array(select_objects, dtype=np.int64))]
# df_fixed_mask = df_cut.reset_index(drop=True)
# df_fixed_mask



im_path = path + 'data/'
psf_path = path + 'psf/'

# noise_path = path + 'rms/'
# noise_type = 'NOISE_MAP'
# band_list = ['r']
# obj_name_location = 1
# deltaPix = 0.1857
# psf_upsample_factor = 2

noise_path = path + 'psf/'
noise_type = 'EXPTIME'
band_list = ['g','r','i']
obj_name_location = 0
deltaPix = 0.27
psf_upsample_factor = 1

if not exists(results_path + '/mosaic_plots'):
    os.mkdir(results_path + '/mosaic_plots')



#Model Lists
lens_model_list = ['SIE','SHEAR'] 
source_model_list = ['SERSIC_ELLIPSE']
lens_light_model_list = ['SERSIC_ELLIPSE']

includeShear = True
use_mask = True
masks_path = csv_path + 'masks'
Mask_rad_file = None#csv_path + 'full_results_sorted.csv' #'/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/results_mask_final/full_results_sorted.csv' #path to csv file or 'None'



#Make dataframes from csv files
# df_list = []
# for i,x in enumerate(csv_paths):
#     df = pd.read_csv(x + 'full_results_sorted.csv',delimiter =',')
#     df_list.append(df.loc[:,:])    
    
# df_final = pd.concat(df_list,axis=0,ignore_index=True)
# df_gamma = pd.read_csv(path + 'Sure_Lens/PEMD_lens/results_Ap26/full_results_sorted.csv',delimiter =',')
df_final = pd.read_csv(csv_path + 'full_results_sorted.csv',delimiter =',')
# df_final = pd.read_csv(csv_path + 'full_results.csv',delimiter =',')




# df_final = df_final[(df_gamma['PEMD_lens.gamma'] <= 1.9) | (df_gamma['PEMD_lens.gamma'] >= 2.1)]
# df_final = df_final[(df_gamma['PEMD_lens.gamma'] >= 1.9) & (df_gamma['PEMD_lens.gamma'] <= 2.1)]
# df_final = df_final[(df_final['PEMD_lens.gamma'] <= 1.9) | (df_final['PEMD_lens.gamma'] >= 2.1)]
# df_final = df_final[(df_final['PEMD_lens.gamma'] >= 1.9) & (df_final['PEMD_lens.gamma'] <= 2.1)]
# df_final = df_final.reset_index(drop=True)

obj_names = []
num = []
for j in range(len(df_final)):
#         fn = df_final.loc[j,'FITS filename']
#         obj_names.append(re.findall('\d+', fn)[obj_name_location])
    obj_names.append(df_final.loc[j,'ID'])
    num.append(df_final.loc[j,'Image num'])
    
# print(num)
    
kwargs_result = df_2_kwargs_results(df = df_final,band_list = band_list, lens_model_list = lens_model_list,
                                   source_model_list = source_model_list,lens_light_model_list = lens_light_model_list)

if fixed_mask_path != None:
    kwargs_result_mask = df_2_kwargs_results(df = df_fixed_mask,band_list = band_list, lens_model_list = lens_model_list,
                                   source_model_list = source_model_list,lens_light_model_list = lens_light_model_list)
# for k,x in enumerate(kwargs_result):
#     print('Object: {}'.format(obj_names[k]))
#     print('Lens: {}'.format(x['kwargs_lens']))
#     print('Source: {}'.format(x['kwargs_source']))
#     print('Lens Light: {}'.format(x['kwargs_lens_light']))
#     print('\n')

im_files = [f for f in listdir(im_path) if isfile('/'.join([im_path,f]))]
psf_files,noise_files = [],[]
psf_files_dict, noise_files_dict = {},{}

for b in band_list: 
    psf_files.append([f for f in listdir(psf_path + '/' + b) if isfile('/'.join([psf_path + '/' + b,f]))])
    noise_files.append([f for f in listdir(noise_path + '/' + b) if isfile('/'.join([noise_path + '/' + b,f]))])
    psf_files_dict[b] = [f for f in listdir(psf_path + '/' + b) if isfile('/'.join([psf_path + '/' + b,f]))]
    noise_files_dict[b] = [f for f in listdir(noise_path + '/' + b) if isfile('/'.join([noise_path + '/' + b,f]))]
    
    
data_pairs_dicts = []
for i,obj in enumerate(obj_names):
#     print(obj)
    for x in im_files:
        if int(obj) == int(re.findall('\d+', x)[obj_name_location]): im = x
            
    psf = {}
    for b in band_list:
        #psf[b] = []
        for file in psf_files_dict[b]:
            if int(obj) == int(re.findall('\d+', file)[obj_name_location]): psf[b] = '/'.join([b,file])
    
    noise = {}
    for b in band_list:
        #noise[b] = []
        for file in noise_files_dict[b]:
            if int(obj) == int(re.findall('\d+', file)[obj_name_location]): noise[b]= '/'.join([b,file])
    if everything_in_df:
        RA = df_final['RA'][df_final['ID'] == int(obj)].to_numpy()[0]
        DEC = df_final['DEC'][df_final['ID'] == int(obj)].to_numpy()[0]
        name = df_final['name'][df_final['ID'] == int(obj)].to_numpy()[0]
    else:
        if ra_dec_path != None:
            df_info = pd.read_csv(ra_dec_path,delimiter =',')
    #         print(type(obj))
    #         RA = df_info['ra'][(df_info[id_col_name] / 1e9).astype('int64')*1e9 == int(obj / 1e9) * 1e9 ].to_numpy()
    #         DEC = df_info['dec'][(df_info[id_col_name] / 1e9).astype('int64')*1e9 == int(obj / 1e9) * 1e9 ].to_numpy()
            RA = df_info['ra'][df_info[id_col_name] == int(obj)].to_numpy()
            DEC = df_info['dec'][df_info[id_col_name] == int(obj)].to_numpy()
            if len(RA) == 0:
                RA = df_info['ra'][df_info['id_final'] == int(obj)].to_numpy()
                DEC = df_info['dec'][df_info['id_final'] == int(obj)].to_numpy()
            RA = RA[0]
            DEC = DEC[0]

    #         for j in range(len(df_info.loc[:,:])):
    #             if int(df_info.loc[j,'id']) == int(obj): RA,DEC = df_info.loc[j,'ra'],df_info.loc[j,'dec']
        else: RA, DEC = 'N/A','N/A'


        if names_path != None:
            df_names = pd.read_csv(names_path,delimiter =',')
            name = df_names['name'][(df_names['ra'] == RA) & (df_names['dec'] == DEC)].to_numpy()[0]
        elif (RA != 'N/A') & (DEC != 'N/A'):
            name = '{} {:.3f}-{:.3f}'.format(name_root,RA,DEC) 
        else: name = obj
        
    data_pairs_dicts.append({'image_data': im , 'psf': psf , 'noise_map': noise, 
                             'noise_type': noise_type,'object_ID': obj, 'image number': num[i],
                             'RA': RA, 'DEC': DEC,'image name': name})

data_pairs_dicts = sorted(data_pairs_dicts, key=lambda k: float(k['object_ID']))

# count = 0
for i,x in enumerate(data_pairs_dicts): 
    if (not x['psf']) or (not x['noise_map']): #or (not x['LRG_data']) or (not x['source_data']):
        continue
#     count += 1
    print(x['image number'])
#     print('image {}'.format(count))
    print('ID: {}'.format(x['object_ID']))
    print('Name: {}'.format(x['image name']))
    print('RA: {}, DEC: {}'.format(x['RA'],x['DEC']))
    print('Full Image data: ',x['image_data'])
#     print('LRG data: ',x['LRG_data'])
#     print('Lensed source data: ',x['source_data'])
    print('PSF: ',x['psf'])
    print('Noise: ',x['noise_map'])
    print('\n')

ncols = 6
nrows = 5
# nrows = len(data_pairs_dicts) + 9

multi_source_model_list = []
multi_lens_light_model_list = []
for b in band_list:
    multi_source_model_list.extend(source_model_list)
    multi_lens_light_model_list.extend(lens_light_model_list)
# multi_source_model_list  

  
lens_class = LensModel(lens_model_list=lens_model_list)
source_class = LightModel(light_model_list = source_model_list)
lens_light_class = LightModel(light_model_list = lens_light_model_list)
# source_class = LightModel(light_model_list = multi_source_model_list)
# lens_light_class = LightModel(light_model_list = multi_lens_light_model_list)

for l in range(len(band_list)):
    f, axes = plt.subplots(nrows,ncols, figsize=(ncols*2,nrows*2.25), sharex=False, sharey=False)
    
#     f, axes = plt.subplots(int(len(data_pairs_dicts) / 6) + 1, 6, figsize=(20,20), sharex=False, sharey=False)
#     f.subplots_adjust(left=0.01, bottom=0.01,wspace=0.1, hspace=0.3)
    f.subplots_adjust(left=0, bottom=0,wspace=0, hspace=0)
#     f.subplots_adjust(wspace=0.1, hspace=0.5)
#     f.subplots_adjust(left=0.0, bottom=0.0, right=0.01, top=0.01, wspace=0.01, hspace=0.01)
    #f, axes = plt.subplots(1, 2, figsize=(20,20), sharex=False, sharey=False)
#     axes = axes.ravel()
    count = 0
    
    
    for it in range(len(data_pairs_dicts[:nrows])):
#     for it in range(len(data_pairs_dicts[:1])):
        it += 27
        print('\n')
        print(data_pairs_dicts[it]['image number'] + '({})'.format(band_list[l]))
        
#         if names_path == None:
#             name = 'CFIS_{:.3f}_{:.3f}'.format(data_pairs_dicts[it]['RA'],data_pairs_dicts[it]['DEC']) 
            
        print('name: ' + data_pairs_dicts[it]['image name'])
        
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

        #     print('calculating background values')
        #     print('\n')
        with HiddenPrints():
            background_rms = calcBackgroundRMS(data_dict['image_data']) #calculate rms background
        #     print('\n')

        lens_info = []

        for i,x in enumerate(data_dict['image_data']):

            lens_info.append({'deltaPix': deltaPix ,
                             'numPix': len(x),
                             'background_rms': background_rms[i],
                             'psf_type': 'PIXEL',
                             'psf_upsample_factor': psf_upsample_factor})

            if noise_type == 'EXPTIME': 
                lens_info[i]['exposure_time'] = data_dict['noise_hdr'][i][0]['EXPTIME']
#                 lens_info[i]['exposure_time'] = 800.
                lens_info[i]['noise_map'] = None
            else:
                lens_info[i]['exposure_time'] = None
                lens_info[i]['noise_map'] = data_dict['noise_map'][i]

        with HiddenPrints():
            kwargs_data, kwargs_psf = prepareData(lens_info,data_dict['image_data'],
                                               data_dict['psf']) 
        mask_list = []
        for b in band_list: 
            with open(masks_path + '/{}/{}.pickle'.format(b,data_pairs_dicts[it]['object_ID']), 'rb') as handle:
#             with open(results_path + '/masks/{}/{}-{}.pickle'.format(b,data_pairs_dicts[it]['image number'],
#                                                                      data_pairs_dicts[it]['object_ID']), 'rb') as handle:
#                 mask = pickle.load(handle)
#                 mask_list.append(mask)
                mask_dict = pickle.load(handle)
                mask_list.append(mask_dict['mask'])


        #prepare fitting kwargs
        kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list,kwargs_constraints = prepareFit(kwargs_data, 
                                                                                         kwargs_psf,
                                                                                         lens_model_list, source_model_list,
                                                                                         lens_light_model_list, 
                                                                                         image_mask_list = mask_list)  

        with HiddenPrints():
            modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result[it], 
                                  arrow_size=0.02, cmap_string="cubehelix",
                                  likelihood_mask_list= mask_list)

        #Calculate Chi^2
#         n_data = modelPlot._imageModel.num_data_evaluate
#         logL = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **kwargs_result[it])
#         red_X_squared = np.abs(logL * 2.0 / n_data)

        model, error_map, cov_param, param = modelPlot._imageModel.image_linear_solve(inv_bool=True, **kwargs_result[it])

#         with HiddenPrints():
#             model_band = ModelBand(multi_band_list, kwargs_model, model[l], error_map[l], cov_param[l],
#                                                param[l], deepcopy(kwargs_result[it]),
#                                                image_likelihood_mask_list=mask_list, band_index=l)


        #print(model_band._reduced_x2)
        
        im_data = ImageData(**kwargs_data[l])
        psf_data = PSF(**kwargs_psf[l])
        im_sim = ImageModel(im_data,psf_data,lens_model_class=lens_class, 
                        source_model_class=source_class, lens_light_model_class= None)
        numPix = lens_info[l]['numPix']
        deltaPix = lens_info[l]['deltaPix']
        
        surfaceBrightness = im_sim.source_surface_brightness([kwargs_result[it]['kwargs_source'][l]],
                                                                 kwargs_lens=kwargs_result[it]['kwargs_lens'],
                                                                 de_lensed= True)
        print(surfaceBrightness)
        print(surfaceBrightness.min(),surfaceBrightness.max())
        print('\n')
        center = None
        source, coords_source = modelPlot.source(numPix = 100, deltaPix = 0.05,band_index=l)
        
        model_band_plot = modelPlot._band_plot_list[l]
        v_min_default = model_band_plot._v_min_default
        v_max_default = model_band_plot._v_max_default
        
#         source = deepcopy(surfaceBrightness)
        
        print(source)
        print(source.min(),source.max())
        fontsize = 10
#         text = ['{}'.format(data_pairs_dicts[it]['object_ID']),
#                 'Reconstructed','Residuals','Convergence','Source',"Magnification"]
#         text = ['{}'.format(data_pairs_dicts[it]['object_ID']),
#                 'Reconstructed','Residuals','Convergence','Source','Deflection']
        
        text = ['{}'.format(data_pairs_dicts[it]['image name']),
                'Reconstructed','Residuals', 'Lens Light','Lensed Source','Source']
        if count >0: #(it != 0) & (it != 8) & (it != 16) & (it != 24): 
            text = ['{}'.format(data_pairs_dicts[it]['image name'])] + [None for i in range(len(text)-1)]
        
        cbar_label=r'log$_{10}$ flux' 
        res_bar_label =r'(f${}_{\rm model}$ - f${}_{\rm data}$)/$\sigma$'
        k_bar_label = r'$\log_{10}\ \kappa$'
        scale_bar_label= True
        if count >0: 
            cbar_label = None
            res_bar_label = None
            k_bar_label = None
            scale_bar_label=False
        
        
        plot_data(model_band_plot,ax=axes[count,0],data = data_dict['image_data'][l],
                          text=text[0],
                          font_size=fontsize,cb_tick_size=fontsize,cut_val = -2,
                          no_arrow = True,colorbar_label=cbar_label,scale_bar_label=scale_bar_label)
        
#         print(len(plt.gca().images))
        
#         cbar = plt.gca().images[-1].colorbar
#         cbar.ax.tick_params(labelsize='large')
        
        
        
        plot_data(model_band_plot,ax=axes[count,1],data = model[l],
                          text=text[1],
                          font_size=fontsize,cut_val = -2,cb_tick_size=fontsize,
                          no_arrow = True,colorbar_label=cbar_label,scale_bar_label=scale_bar_label)
        
        ra_crit_list, dec_crit_list = model_band_plot._critical_curves()
        
        plot_line_set(axes[count,1], model_band_plot._coords, ra_crit_list, dec_crit_list, color='r')
        
         #plot residuals
        normalized_residual_plot(model_band_plot,ax=axes[count,2], v_min=-6, v_max=6, 
                                           text = text[2],
                                           font_size=fontsize,cb_tick_size=fontsize,
                                           cmap = "coolwarm",no_arrow = True,colorbar_label=res_bar_label,
                                scale_bar_label=scale_bar_label)
        
        
        
        #plot convergence
#         convergence_plot(model_band_plot,ax=axes[count, 3], v_max=1,
#                  font_size = fontsize,cb_tick_size=fontsize,
#                  text=text[3],no_arrow = True,colorbar_label=k_bar_label,
#                         scale_bar_label=scale_bar_label)

        lens_light = model_band_plot._bandmodel.image(model_band_plot._kwargs_lens_partial, 
                                                         model_band_plot._kwargs_source_partial, 
                                                         model_band_plot._kwargs_lens_light_partial,
                                                         model_band_plot._kwargs_ps_partial, 
                                                         unconvolved=True, source_add=False,
                                                         lens_light_add=True, point_source_add=False)
    
        plot_data(model_band_plot,ax=axes[count,3],data = lens_light,
                          text=text[3],
                          font_size=fontsize,cut_val = -2,cb_tick_size=fontsize,
                          no_arrow = True,colorbar_label=cbar_label,scale_bar_label=scale_bar_label)
    
        
        lensed_source = model_band_plot._bandmodel.image(model_band_plot._kwargs_lens_partial, 
                                                         model_band_plot._kwargs_source_partial, 
                                                         model_band_plot._kwargs_lens_light_partial,
                                                         model_band_plot._kwargs_ps_partial, 
                                                         unconvolved=True, source_add=True,
                                                         lens_light_add=False, point_source_add=False)
    
        plot_data(model_band_plot,ax=axes[count,4],data = lensed_source,
                          text=text[4],
                          font_size=fontsize,cut_val = -2,cb_tick_size=fontsize,
                          no_arrow = True,colorbar_label=cbar_label,scale_bar_label=scale_bar_label)
        
        
        log_source = np.log10(source)
        log_source[np.isnan(log_source)] = -5
        v_min = max(np.min(log_source), -5)
        v_max = min(np.max(log_source), 10)
        source[source < 10**(v_min)] = 10**(v_min) # to remove weird shadow in plot
        center_source = [kwargs_result[it]['kwargs_source'][l]['center_x'],kwargs_result[it]['kwargs_source'][l]['center_y']]
        
#         model_band_plot.source_plot(ax=axes[count,4], v_min = v_min,text=text[4],center = center_source,
#                                     numPix=100, deltaPix_source=0.05,scale_size=0.5, with_caustics=True,no_arrow = True)
        
        source_plot(model_band_plot,ax=axes[count,5], v_min = v_min,text=text[5],center = center_source,
                                    numPix=100, deltaPix_source=0.05,scale_size=0.5, with_caustics=True,
                                    cb_tick_size=fontsize,font_size= fontsize, no_arrow = True,colorbar_label=cbar_label,
                                    scale_bar_label=scale_bar_label)
    
        
       
        
#         plot_data(model_band_plot,ax=axes[count,4],data = source,v_min = -2,text=text[4],
#                   font_size=fontsize,cut_val = -5,cb_tick_size=fontsize,no_arrow = True)
#         ra_caustic_list, dec_caustic_list = model_band_plot._caustics()
#         plot_util.plot_line_set(axes[count,4], model_band_plot._coords, ra_caustic_list, dec_caustic_list, color='b')
        
#         modelPlot.deflection_plot(ax=axes[it, 4],band_index=l,axis = 0,with_caustics=True,font_size = fontsize,text=text[5])
#         model_band_plot._cmap = "coolwarm"
#         magnification_plot(model_band_plot,ax=axes[count, 4],
#                            font_size = fontsize,cb_tick_size=fontsize,
#                            text=text[4],no_arrow = True)
        


        
        
#         lens_plot.lens_model_plot(axes[it, 4], lensModel=lens_class, kwargs_lens=kwargs_result[it]['kwargs_lens'], 
#                                   numPix=numPix, deltaPix=deltaPix,
#                                   sourcePos_x=kwargs_result[it]['kwargs_source'][l]['center_x'], 
#                                   sourcePos_y=kwargs_result[it]['kwargs_source'][l]['center_y'], 
#                                   point_source=False,coord_inverse=True, with_caustics=True)
        
#         plot_util.text_description(axes[it, 4], d_s, text="Critical Lines", color="w", backgroundcolor='k',
#                          flipped=False, font_size=15)
        
#         axes[it,2].set_title('Image: {}'.format(it+1),fontsize=25)
#         axes[it,3].set_title('ID: {}'.format(obj_names[it]),fontsize=25)
#         axes[it,5].set_title('$ \chi^2 $ (all): {:.4f} \n $\chi^2$({} band){:.4f}'
#                                            .format(red_X_squared,band_list[l], 
#                                            model_band._reduced_x2),fontsize=20)
        if 'PEMD' in lens_model_list:
            axes[count].text(len(mask_dict['mask'])/2, 1, '$\gamma$ = {:.4f}'.format(kwargs_result[it]['kwargs_lens'][0]['gamma']), horizontalalignment='center',fontsize=25,color='r')
            
        count += 1
        
        if (fixed_mask_path != None) & (count < nrows):
            if str(data_pairs_dicts[it]['object_ID']) in select_objects:
                
                print('Fixed mask plot for object: {}'.format(data_pairs_dicts[it]['object_ID']))
                index = select_objects.index(str(data_pairs_dicts[it]['object_ID']))
                mask_list = []
                for b in band_list: 
                    alt_masks_path = fixed_mask_path + 'masks'
                    with open(alt_masks_path + '/{}/{}.pickle'.format(b,data_pairs_dicts[it]['object_ID']), 'rb') as handle:
                        mask_dict = pickle.load(handle)
                        mask_list.append(mask_dict['mask'])
                #prepare fitting kwargs
                kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list,kwargs_constraints = prepareFit(kwargs_data, 
                                                                                                 kwargs_psf,lens_model_list,
                                                                                                 source_model_list,
                                                                                                 lens_light_model_list, 
                                                                                                 image_mask_list = mask_list)
                with HiddenPrints():
                    modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result_mask[index], 
                                          arrow_size=0.02, cmap_string="cubehelix",
                                          likelihood_mask_list= mask_list)
                model, error_map, cov_param, param = modelPlot._imageModel.image_linear_solve(inv_bool=True, 
                                                                                              **kwargs_result_mask[index])
                source, coords_source = modelPlot.source(numPix = 100, deltaPix = 0.05,band_index=l)
                model_band_plot = modelPlot._band_plot_list[l]
                
                text = ['{}'.format(data_pairs_dicts[it]['image name']),'Reconstructed','Residuals', 'Convergence','Source']
                if count >0: #(it != 0) & (it != 8) & (it != 16) & (it != 24): 
                    text = ['{}'.format(data_pairs_dicts[it]['image name'])] + [None for i in range(len(text)-1)]
                
                
                cbar_label=r'log$_{10}$ flux' 
                res_bar_label =r'(f${}_{\rm model}$ - f${}_{\rm data}$)/$\sigma$'
                k_bar_label = r'$\log_{10}\ \kappa$'
                scale_bar_label= True
                if count >0: 
                    cbar_label = None
                    res_bar_label = None
                    k_bar_label = None
                    scale_bar_label=False
                
                #plot data
                plot_data(model_band_plot,ax=axes[count,0],data = data_dict['image_data'][l],
                          text=text[0],
                          font_size=fontsize,cb_tick_size=fontsize,cut_val = -2,
                          no_arrow = True,colorbar_label=cbar_label,
                          scale_bar_label=scale_bar_label)
                #plot reconstruction
                plot_data(model_band_plot,ax=axes[count,1],data = model[l],
                          text=text[1],
                          font_size=fontsize,cut_val = -2,cb_tick_size=fontsize,
                          no_arrow = True,colorbar_label=cbar_label,scale_bar_label=scale_bar_label)
                ra_crit_list, dec_crit_list = model_band_plot._critical_curves()
                plot_line_set(axes[count,1], model_band_plot._coords, ra_crit_list, dec_crit_list, color='r')
                
                #plot residuals
                normalized_residual_plot(model_band_plot,ax=axes[count,2], v_min=-6, v_max=6, 
                                                   text = text[2],
                                                   font_size=fontsize,cb_tick_size=fontsize,
                                                   cmap = "coolwarm",no_arrow = True,colorbar_label=res_bar_label,
                                        scale_bar_label=scale_bar_label)
                #plot convergence
                convergence_plot(model_band_plot,ax=axes[count, 3], v_max=1,
                         font_size = fontsize,cb_tick_size=fontsize,
                         text=text[3],no_arrow = True,colorbar_label=k_bar_label,scale_bar_label=scale_bar_label)
                
                #plot source
                center_source = [kwargs_result_mask[index]['kwargs_source'][l]['center_x'],
                                 kwargs_result_mask[index]['kwargs_source'][l]['center_y']]
                
                
                
                source_plot(model_band_plot,ax=axes[count,4], v_min = v_min,text=text[4],center = center_source,
                                            numPix=100, deltaPix_source=0.05,scale_size=0.5, with_caustics=True,
                                            cb_tick_size=fontsize,font_size= fontsize, no_arrow = True,colorbar_label=cbar_label,
                                            scale_bar_label=scale_bar_label)
                
                
                count += 1
            
#         rect = plt.Rectangle(
#         # (lower-left corner), width, height
#         (0.003, 0.59),0.994, 0.405, fill=False, color="red", lw=1.5, ls='--',
#         zorder=1000, transform=f.transFigure, figure=f,rasterized=True
#         )
#         f.patches.extend([rect])
        
#         rect2 = plt.Rectangle(
#         # (lower-left corner), width, height
#         (0.003, 0.005),0.994, 0.39, fill=False, color="red", lw=1.5, ls='--',
#         zorder=1000, transform=f.transFigure, figure=f,rasterized=True
#         )
#         f.patches.extend([rect2])
        
#         rect3 = plt.Rectangle(
#         # (lower-left corner), width, height
#         (0.003, 0.655),0.994, 0.34, fill=False, color="red", lw=1.5, ls='--',
#         zorder=1000, transform=f.transFigure, figure=f,rasterized=True
#         )
#         f.patches.extend([rect3])
        
#         rect4 = plt.Rectangle(
#         # (lower-left corner), width, height
#         (0.003, 0.33),0.994, 0.325, fill=False, color="red", lw=1.5, ls='--',
#         zorder=1000, transform=f.transFigure, figure=f,rasterized=True
#         )
#         f.patches.extend([rect4])
        
#         rect5 = plt.Rectangle(
#         # (lower-left corner), width, height
#         (0.003, 0.005),0.994, 0.325, fill=False, color="red", lw=1.5, ls='--',
#         zorder=1000, transform=f.transFigure, figure=f,rasterized=True
#         )
#         f.patches.extend([rect5])
        
        
        
        print(count)
        if count >= nrows:
            break

                
                

    
#     for i,a in enumerate(axes):
#         if i > count:#len(data_pairs_dicts):
#             a.set_axis_off()
            
    f.tight_layout()
    f.savefig(results_path + '/mosaic_plots/{}_band_results_28to32.pdf'.format(band_list[l]),dpi = 200)
    f.savefig(results_path + '/mosaic_plots/{}_band_results_28to32.png'.format(band_list[l]),dpi = 100)
    plt.close(f)
    
    