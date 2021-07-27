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

# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/'
# csv_path = path + 'results_mask_final/'
# masks_path = path + 'results_mask_final/masks'
# results_path = path + 'results_mask_final/residual_plots' 

path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/'
csv_path = path + 'muscadet_deblended/results_long_deblended/'
masks_path = path + 'muscadet_deblended/results_long_deblended/masks'
results_path = path + 'muscadet_deblended/results_long_deblended'
# csv_path = path + 'SIE_lens/results_Ap30/'
# masks_path = path + 'SIE_lens/results_Ap30/masks'
# results_path = path + 'SIE_lens/results_Ap30'

# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/ringcatalog/'
# csv_path = path + 'results_May3/'
# results_path = path + 'results_May3'

# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_lenses/'
# csv_path = path + 'Sure_Lens/SIE_lens/results_Ap30/'
# masks_path = path + 'Sure_Lens/SIE_lens/results_Ap30/masks'
# results_path = path + 'Sure_Lens/SIE_lens/results_Ap30'

# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_deblended/deblended_image_2/modeling_results_normal/'
# im_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_deblended/deblended_image_2/originals/lenses/'
# psf_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_deblended/deblended_image_2/psf/'
# csv_path = path + 'results_model_orig/'
# results_path = path + 'results_model_orig'

skip_obj = ['2812456','3310601','9322010']

im_path = path + 'data/'
psf_path = path + 'psf/'
# noise_path = psf_path
# noise_path = path + 'rms/'
# noise_type = 'NOISE_MAP'
noise_path = path + 'psf/'
noise_type = 'EXPTIME'

if not exists(results_path + '/residual_plots'):
    os.mkdir(results_path + '/residual_plots')

#Folder names for data, psf, noise map, original image [TO DO BY USER]
band_list = ['g','r','i']
obj_name_location = 0
deltaPix = 0.27
psf_upsample_factor = 1

# band_list = ['r']
# obj_name_location = 1
# deltaPix = 0.1857
# psf_upsample_factor = 2

ncols = 17

#Model Lists
lens_model_list = ['SIE','SHEAR'] 
source_model_list = ['SERSIC_ELLIPSE']
lens_light_model_list = ['SERSIC_ELLIPSE']

includeShear = True
use_mask = True
masks_path = csv_path + 'masks'
Mask_rad_file = None#csv_path + 'full_results_sorted.csv' #'/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/results_mask_final/full_results_sorted.csv' #path to csv file or 'None'

id_col_name = 'ID'

#Make dataframes from csv files
# df_list = []
# for i,x in enumerate(csv_paths):
#     df = pd.read_csv(x + 'full_results_sorted.csv',delimiter =',')
#     df_list.append(df.loc[:,:])    
    
# df_final = pd.concat(df_list,axis=0,ignore_index=True)
# df_gamma = pd.read_csv(path + 'Sure_Lens/PEMD_lens/results_Ap26/full_results_sorted.csv',delimiter =',')
df_final = pd.read_csv(csv_path + 'full_results_sorted.csv',delimiter =',')

# df_final = df_final[(df_gamma['PEMD_lens.gamma'] <= 1.9) | (df_gamma['PEMD_lens.gamma'] >= 2.1)]
# df_final = df_final[(df_gamma['PEMD_lens.gamma'] >= 1.9) & (df_gamma['PEMD_lens.gamma'] <= 2.1)]
# df_final = df_final[(df_final['PEMD_lens.gamma'] <= 1.9) | (df_final['PEMD_lens.gamma'] >= 2.1)]
# df_final = df_final[(df_final['PEMD_lens.gamma'] >= 1.9) & (df_final['PEMD_lens.gamma'] <= 2.1)]
df_final = df_final.reset_index(drop=True)

obj_names = []
num = []
for j in range(len(df_final)):
#         fn = df_final.loc[j,'FITS filename']
#         obj_names.append(re.findall('\d+', fn)[obj_name_location])
    obj_names.append(df_final.loc[j,'ID'])
    num.append(df_final.loc[j,'Image num'])
    
print(num)
    
kwargs_result = df_2_kwargs_results(df = df_final,band_list = band_list, lens_model_list = lens_model_list,
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

    data_pairs_dicts.append({'image_data': im , 'psf': psf , 'noise_map': noise, 
                             'noise_type': noise_type,'object_ID': obj, 'image number': num[i]})
    

data_pairs_dicts = sorted(data_pairs_dicts, key=lambda k: float(k['object_ID']))

# count = 0
for i,x in enumerate(data_pairs_dicts): 
    if (not x['psf']) or (not x['noise_map']): #or (not x['LRG_data']) or (not x['source_data']):
        continue
#     count += 1
    print(x['image number'])
#     print('image {}'.format(count))
    print('ID: {}'.format(x['object_ID']))
#     print('RA: {}, DEC: {}'.format(x['RA'],x['DEC']))
    print('Full Image data: ',x['image_data'])
#     print('LRG data: ',x['LRG_data'])
#     print('Lensed source data: ',x['source_data'])
    print('PSF: ',x['psf'])
    print('Noise: ',x['noise_map'])
    print('\n')



# nrows = ceil(len(data_pairs_dicts) / ncols)
nrows = ceil(49 / ncols)

for l in range(len(band_list)):
    f, axes = plt.subplots(nrows,ncols, figsize=(ncols*5,nrows*5 + 5), sharex=False, sharey=False)
    
#     f, axes = plt.subplots(int(len(data_pairs_dicts) / 6) + 1, 6, figsize=(20,20), sharex=False, sharey=False)
#     f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
    #f, axes = plt.subplots(1, 2, figsize=(20,20), sharex=False, sharey=False)
    axes = axes.ravel()
    count = 0
    for it in range(len(data_pairs_dicts)):
        if str(data_pairs_dicts[it]['object_ID']) in skip_obj:
            continue
    #     it = 0
        print('\n')
        print(data_pairs_dicts[it]['image number'] + '({})'.format(band_list[l]))
        print(str(data_pairs_dicts[it]['object_ID']) + '({})'.format(band_list[l]))
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
#             with open(results_path + '/masks/{}/Image_{}-{}.pickle'.format(b,it+1,data_pairs_dicts[it]['object_ID']), 'rb') as handle:
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
                                  arrow_size=0.02, cmap_string="gist_heat",
                                  likelihood_mask_list= mask_list)

        #Calculate Chi^2
        n_data = modelPlot._imageModel.num_data_evaluate
        logL = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **kwargs_result[it])
        red_X_squared = np.abs(logL * 2.0 / n_data)

        model, error_map, cov_param, param = modelPlot._imageModel.image_linear_solve(inv_bool=True, **kwargs_result[it])

        with HiddenPrints():
            model_band = ModelBand(multi_band_list, kwargs_model, model[l], error_map[l], cov_param[l],
                                               param[l], deepcopy(kwargs_result[it]),
                                               image_likelihood_mask_list=mask_list, band_index=l)


        #print(model_band._reduced_x2)


        modelPlot.normalized_residual_plot(ax=axes[count], v_min=-6, v_max=6, 
#                                            text = None,
#                                            text=r'$\gamma$ = {:.4f}'.format(kwargs_result[it]['kwargs_lens'][0]['gamma']),
                                           font_size=0,band_index=l) 
        
        axes[count].set_title('$ID:$ {} \n $ \chi^2 $ (all): {:.4f} \n $\chi^2$({} band){:.4f}'
                                           .format(obj_names[it], red_X_squared,band_list[l], 
                                           model_band._reduced_x2),fontsize=25)
        if 'PEMD' in lens_model_list:
            axes[count].text(len(mask_dict['mask'])/2, 1, '$\gamma$ = {:.4f}'.format(kwargs_result[it]['kwargs_lens'][0]['gamma']), horizontalalignment='center',fontsize=25,color='r')
#         len(color_img)/5

        count += 1
    
    for i,a in enumerate(axes):
        if i >= count:
            a.set_axis_off()
            
    f.tight_layout()
    f.savefig(results_path + '/residual_plots/{}_band_residuals_wide.png'.format(band_list[l]),dpi = 100)
    plt.close(f)
    
    