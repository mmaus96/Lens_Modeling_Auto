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
from Lens_Modeling_Auto.auto_modeling_functions import openFITS
from Lens_Modeling_Auto.auto_modeling_functions import find_components
from Lens_Modeling_Auto.auto_modeling_functions import calcBackgroundRMS
from Lens_Modeling_Auto.auto_modeling_functions import prepareData
from Lens_Modeling_Auto.auto_modeling_functions import prepareFit
from Lens_Modeling_Auto.auto_modeling_functions import find_components
from Lens_Modeling_Auto.auto_modeling_functions import mask_for_sat
from Lens_Modeling_Auto.auto_modeling_functions import mask_for_lens_gal
from Lens_Modeling_Auto.auto_modeling_functions import df_2_kwargs_results
from Lens_Modeling_Auto.plot_functions import make_modelPlots
from functools import reduce
from matplotlib.colors import SymLogNorm
import re
from matplotlib.patches import Circle
from copy import deepcopy
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Analysis.image_reconstruction import ModelBand


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_lenses/'
# csv_paths = [path + 'results_new_priors/']
# results_path = path + 'results_new_priors'

# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/ringcatalog/'
# csv_paths = [path + 'results_new_priors/']
# results_path = path + 'results_new_priors'

path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Sure_Lens/'
csv_paths = [path + 'results_new_priors/']
results_path = path + 'results_new_priors'

modelPlot_path = results_path + '/modelPlot_results_remake' 

im_path = path + 'data/'
psf_path = path + 'psf/'
noise_path = path + 'psf/'
noise_type = 'EXPTIME'

# im_path = path + 'data/'
# psf_path = path + 'psf/'
# noise_path = path + 'rms/'
# noise_type = 'NOISE_MAP'

if not exists(results_path):
    os.mkdir(results_path)
if not exists(modelPlot_path):
    os.mkdir(modelPlot_path)

#Folder names for data, psf, noise map, original image [TO DO BY USER]
band_list = ['g','r','i']
obj_name_location = 0
deltaPix = 0.27
psf_upsample_factor = 1

# band_list = ['r']
# obj_name_location = 1
# deltaPix = 0.1857
# psf_upsample_factor = 2

zeroPt = 30
# numCores = 1
includeShear = True
use_mask = True

#Make dataframes from csv files
df_list = []
for i,x in enumerate(csv_paths):
    df = pd.read_csv(x + 'full_results.csv',delimiter =',')
    df_list.append(df.loc[1:,:])    
    
df_final = pd.concat(df_list,axis=0,ignore_index=True)
df_final['Unnamed: 0'] = df_final['Unnamed: 0.1']
df_final = df_final.drop('Unnamed: 0.1',axis=1)


obj_names = []
im_num = []
for j in range(len(df_final)):
        fn = df_final.loc[j,'Unnamed: 1']
        im_num.append(df_final.loc[j,'Unnamed: 0'])
        obj_names.append(re.findall('\d+', fn)[obj_name_location])
        
        


kwargs_result = df_2_kwargs_results(df = df_final,band_list = band_list,ignore_1st_line = False,includeShear = True)

for k,x in enumerate(kwargs_result):
    print('Object: {}'.format(obj_names[k]))
    print('Lens: {}'.format(x['kwargs_lens']))
    print('Source: {}'.format(x['kwargs_source']))
    print('Lens Light: {}'.format(x['kwargs_lens_light']))
    print('\n')

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
    for x in im_files:
        if obj in x: im = x
            
    psf = {}
    for b in band_list:
        #psf[b] = []
        for i,file in enumerate(psf_files_dict[b]):
            if obj in file: psf[b] = '/'.join([b,file])
    
    noise = {}
    for b in band_list:
        #noise[b] = []
        for i,file in enumerate(noise_files_dict[b]):
            if obj in file: noise[b]= '/'.join([b,file])

    data_pairs_dicts.append({'image_data': im , 'psf': psf , 'noise_map': noise, 'noise_type': noise_type, 'object_ID': obj})
    
    
    
# for l,b in enumerate(band_list):
#     f, axes = plt.subplots(int(len(data_pairs_dicts) / 6) + 1, 6, figsize=(20,20), sharex=False, sharey=False)
#     f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.)
    #f, axes = plt.subplots(1, 2, figsize=(20,20), sharex=False, sharey=False)
#     axes = axes.ravel()
band_path = modelPlot_path + '/' + b

if not exists(band_path):
    os.mkdir(band_path)
    
for it in range(len(data_pairs_dicts)):
#     it = 0

    data,hdr = openFITS(im_path + '/' + data_pairs_dicts[it]['image_data'])
    psf, psf_hdr = [],[]
    noise_map,noise_hdr = [],[]
    for z in band_list:
        d,h = openFITS(psf_path  + '/' + data_pairs_dicts[it]['psf'][z])
#         psf.extend(d)
#         psf_hdr.extend(h)
        psf.append(d)
        psf_hdr.append(h)


        d2,h2 = openFITS(noise_path  + '/' + data_pairs_dicts[it]['noise_map'][z])
#         noise_map.extend(d2)
#         noise_hdr.extend(h2)
        noise_map.append(d2)
        noise_hdr.append(h2)


    data_dict = {'image_data': [], 'image_hdr': [], 
                 'psf': psf, 'psf_hdr': psf_hdr, 
                 'noise_map': noise_map, 'noise_hdr': noise_hdr}

    for i,y in enumerate(band_list):
#         data_dict['image_data'].append(data[i])
#         data_dict['image_hdr'].append(hdr[0])
        data_dict['image_data'].append(data[0][i])
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
            lens_info[i]['noise_map'] = None
        else:
            lens_info[i]['exposure_time'] = None
            lens_info[i]['noise_map'] = data_dict['noise_map'][i]

    kwargs_data, kwargs_psf = prepareData(lens_info,data_dict['image_data'],
                                           data_dict['psf']) 

    #Model Lists
    if includeShear == True:
        lens_model_list = ['SIE','SHEAR'] 
    else:
        lens_model_list = ['SIE']

    source_model_list = ['SERSIC_ELLIPSE']
    lens_light_model_list = ['SERSIC_ELLIPSE']

    gal_mask_list = []
    mask_list = []

    for data in kwargs_data: 
        gal_mask_list.append(mask_for_lens_gal(data['image_data'],deltaPix))
        if use_mask:
            mask_list.append(mask_for_sat(data['image_data'],deltaPix))
        else: mask_list = None


    #prepare fitting kwargs
    kwargs_likelihood, kwargs_model, kwargs_data_joint, multi_band_list,kwargs_constraints = prepareFit(kwargs_data, 
                                                                                     kwargs_psf,
                                                                                     lens_model_list, source_model_list,
                                                                                     lens_light_model_list, 
                                                                                     image_mask_list = mask_list)  

#         with HiddenPrints():
#             modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result[it], 
#                                   arrow_size=0.02, cmap_string="gist_heat",
#                                   likelihood_mask_list= mask_list)

#         #Calculate Chi^2
#         n_data = modelPlot._imageModel.num_data_evaluate
#         logL = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **kwargs_result[it])
#         red_X_squared = np.abs(logL * 2.0 / n_data)

#         model, error_map, cov_param, param = modelPlot._imageModel.image_linear_solve(inv_bool=True, **kwargs_result[it])

#         with HiddenPrints():
#             model_band = ModelBand(multi_band_list, kwargs_model, model[l], error_map[l], cov_param[l],
#                                                param[l], deepcopy(kwargs_result[it]),
#                                                image_likelihood_mask_list=mask_list, band_index=l)



    red_X_squared = make_modelPlots(multi_band_list,kwargs_model,kwargs_result[it],
                                kwargs_data,kwargs_psf, lens_info,
                                lens_model_list,source_model_list,lens_light_model_list,
                                mask_list,band_list,modelPlot_path,it+1,data_pairs_dicts[it]['object_ID'])

    #print(model_band._reduced_x2)


#     modelPlot.normalized_residual_plot(ax=axes[it], v_min=-6, v_max=6,
#                                        text='$ID:$ {} \n $ \chi^2 $ (all): {:.4f} \n $\chi^2$({} band){:.4f}'
#                                        .format(obj_names[it], red_X_squared,band_list[l], 
#                                        model_band._reduced_x2),font_size=10,band_index=l) 

#         f, axes = plt.subplots(4, 3, figsize=(20,20), sharex=False, sharey=False)

#     #     band_path = modelPlot_path + '/' + b

#     #     if not exists(band_path):
#     #         os.mkdir(band_path)

#         modelPlot.data_plot(ax=axes[0,0],band_index=l)
#         modelPlot.model_plot(ax=axes[0,1],band_index=l)
#         modelPlot.normalized_residual_plot(ax=axes[0,2], v_min=-6, v_max=6,band_index=l)
#         modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100,band_index=l)
#         modelPlot.convergence_plot(ax=axes[1, 1], v_max=1,band_index=l)
#         modelPlot.magnification_plot(ax=axes[1, 2],band_index=l)
#         modelPlot.decomposition_plot(ax=axes[2,0], text='Lens light', lens_light_add=True, unconvolved=True,band_index=l)
#         modelPlot.decomposition_plot(ax=axes[3,0], text='Lens light convolved', lens_light_add=True,band_index=l)
#         modelPlot.decomposition_plot(ax=axes[2,1], text='Source light', source_add=True, unconvolved=True,band_index=l)
#         modelPlot.decomposition_plot(ax=axes[3,1], text='Source light convolved', source_add=True,band_index=l)
#         modelPlot.decomposition_plot(ax=axes[2,2], text='All components', source_add=True, lens_light_add=True, unconvolved=True,band_index=l)
#         modelPlot.decomposition_plot(ax=axes[3,2], text='All components convolved', source_add=True, lens_light_add=True, point_source_add=True,band_index=l)
#         f.tight_layout()
#         f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
#         f.suptitle('$ID:$ {} \n $ \chi^2 $ (all): {:.4f} \n $\chi^2$({} band){:.4f}'
#                                            .format(obj_names[it], red_X_squared,band_list[l], 
#                                            model_band._reduced_x2),fontsize=30)     

#         f.savefig(band_path + '/{}-{}.png'.format(im_num[it],obj_names[it]),dpi = 200)
#         plt.close(f)