import sys
if sys.platform[:5] == 'linux':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import astropy.io.fits as pyfits
import astropy.io.ascii as ascii
import scipy
import pandas as pd
from scipy.ndimage.filters import gaussian_filter as gauss1D
from scipy import optimize
from pylab import figure, cm
from matplotlib.colors import LogNorm
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Util.param_util import ellipticity2phi_q
from lenstronomy.Util.param_util import shear_cartesian2polar
from os import walk
from os import listdir
from os.path import isfile, join, exists
from Lens_Modeling_Auto.auto_modeling_functions import df_2_dict
from math import ceil
from copy import deepcopy


#### Create dataframe from csv file(s) ####

path_CFIS = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_lenses/Sure_Lens/'
csv_path_CFIS = path_CFIS + 'SIE_lens/results_May31/'
results_path_CFIS = path_CFIS + 'SIE_lens/results_May31'

path_DES = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/'
csv_path_DES = path_DES + 'SIE_lens/results_Jun1/'
results_path_DES = path_DES + 'SIE_lens/results_Jun1'

path_rings = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/Rings_catalog/new_priors/'
csv_path_rings = path_rings + 'results_Jun11/'
results_path_rings = path_rings + 'results_Jun11/'

# if not exists(results_path):
#     os.mkdir(results_path)
    
lens_model_list = ['SIE','SHEAR']
# lens_model_list = ['PEMD','SHEAR']
source_model_list = ['SERSIC_ELLIPSE']
lens_light_model_list = ['SERSIC_ELLIPSE']
# band_list = ['g','r','i']
band_list = ['r']

df_DES = pd.read_csv(csv_path_DES + 'full_results_sorted.csv',delimiter =',')
df_CFIS = pd.read_csv(csv_path_CFIS + 'full_results_sorted.csv',delimiter =',')
df_rings = pd.read_csv(csv_path_rings + 'full_results_sorted.csv',delimiter =',')

dict_results_DES = df_2_dict(df_DES,band_list,lens_model_list,source_model_list,lens_light_model_list)
dict_results_CFIS = df_2_dict(df_CFIS,band_list,lens_model_list,source_model_list,lens_light_model_list)
dict_results_rings = df_2_dict(df_rings,band_list,lens_model_list,source_model_list,lens_light_model_list)

dict_results = [dict_results_DES,dict_results_CFIS,dict_results_rings]
labels = ['DES lenses', 'CFIS lenses', 'ring galaxies']
# print(data_dict['shear']['gamma'])


num_plots = 1 
cols = 9

for prof in lens_model_list:
    results = np.array(list(dict_results[0]['lens'][prof].items()))
    print(len(results))
    num_plots += (len(results) - 2)
    
for band in band_list:
    for prof in source_model_list:
        key = '{} Band: {}'.format(band,prof)
        results = np.array(list(dict_results[0]['source'][key].items()))
        print(len(results))
        num_plots += (len(results) - 2)

    for prof in lens_light_model_list:
        key = '{} Band: {}'.format(band,prof)
        results = np.array(list(dict_results[0]['lens_light'][key].items()))
        print(len(results))
        num_plots += (len(results) - 2)


# for prof in source_model_list:
#     key = '{} Band: {}'.format(band_list[0],prof)
#     results = np.array(list(dict_results['source'][key].items()))
#     print(len(results))
#     num_plots += (len(results) - 2)

# for prof in lens_light_model_list:
#     key = '{} Band: {}'.format(band_list[0],prof)
#     results = np.array(list(dict_results['lens_light'][key].items()))
#     print(len(results))
#     num_plots += (len(results) - 2)
        
print(num_plots)

rows = ceil(num_plots /  cols)

f,axes = plt.subplots(rows,cols, figsize = (cols*5,rows*5))
f.subplots_adjust(hspace=0.5,wspace=0.5)
count = 0
ax = axes.ravel()

ax[0].hist([dict_results[i]['Reduced Chi^2'][dict_results[i]['Reduced Chi^2'] < 5] for i in range(len(dict_results))],
           label=labels,density=True)
ax[0].set_title('Reduced Chi^2')
ax[0].set_xlabel('Value')
ax[0].set_ylabel('Density')
ax[count].legend()

count +=1

for prof in lens_model_list:
    results = [np.array(list(dict_results[i]['lens'][prof].items())) for i in range(len(dict_results))]
    for i in range(len(results[0])):
        if (results[0][i][0] == 'ra_0') or (results[0][i][0] == 'dec_0') or (results[0][i][0] == 'center_x') or (results[0][i][0] == 'center_y'):
            continue
        ax[count].hist([results[j][i][1][dict_results[j]['Reduced Chi^2'] < 1.5] for j in range(len(dict_results))],
                       label=labels,density=True)
        ax[count].set_title('Lens \n {} \n {}'.format(prof,results[0][i][0]))
        ax[count].set_xlabel('Value')
        ax[count].set_ylabel('Density')
        ax[count].legend()
        count += 1
    
for band in band_list:
    for prof in source_model_list:
        key = '{} Band: {}'.format(band,prof)
        results = [np.array(list(dict_results[i]['source'][key].items())) for i in range(len(dict_results))]
        for i in range(len(results[0])):
            if (results[0][i][0] == 'center_x') or (results[0][i][0] == 'center_y'):
                continue
            ax[count].hist([results[j][i][1][dict_results[j]['Reduced Chi^2'] < 1.5] for j in range(len(dict_results))],label=labels,density=True)
            ax[count].set_title('{} Band: Source \n {} \n {}'.format(band,prof,results[0][i][0]))
            ax[count].set_xlabel('Value')
            ax[count].set_ylabel('Density')
            ax[count].legend()
            count += 1

    for prof in lens_light_model_list:
        key = '{} Band: {}'.format(band,prof)
        results = [np.array(list(dict_results[i]['lens_light'][key].items())) for i in range(len(dict_results))]
        for i in range(len(results[0])):
            if (results[0][i][0] == 'center_x') or (results[0][i][0] == 'center_y'):
                continue
            ax[count].hist([results[j][i][1][dict_results[j]['Reduced Chi^2'] < 1.5] for j in range(len(dict_results))],label=labels,density=True)
            ax[count].set_title('{} Band: Lens Light \n {} \n {}'.format(band,prof,results[0][i][0]))
            ax[count].set_xlabel('Value')
            ax[count].set_ylabel('Density')
            ax[count].legend()
            count += 1

# for prof in source_model_list:
#     key0 = '{} Band: {}'.format(band_list[0],prof)
#     source_params = np.array(list(dict_results['source'][key].items()))[:,0]
#     for i,param in enumerate(source_params):
#         if (param == 'center_x') or (param == 'center_y'):
#             continue

#         band_keys = []
#         for band in band_list:
#             key = '{} Band: {}'.format(band,prof)
#             band_keys.append(key)
# #             results = np.array(list(dict_results['source'][key].items()))
#         ax[count].hist([dict_results['source'][key][param][dict_results['Reduced Chi^2'] < 1.5] for key in band_keys],label=[b for b in band_list])
#         ax[count].set_title('Source \n {} \n {}'.format(prof,param))
#         ax[count].set_xlabel('Value')
#         ax[count].set_ylabel('# Occurences')
#         ax[count].legend()
#         count += 1

# for prof in lens_light_model_list:
#     key0 = '{} Band: {}'.format(band_list[0],prof)
#     lens_light_params = np.array(list(dict_results['lens_light'][key].items()))[:,0]
#     for i,param in enumerate(lens_light_params):
#         if (param == 'center_x') or (param == 'center_y'):
#             continue

#         band_keys = []
#         for band in band_list:
#             key = '{} Band: {}'.format(band,prof)
#             band_keys.append(key)
# #             results = np.array(list(dict_results['source'][key].items()))
#         ax[count].hist([dict_results['lens_light'][key][param][dict_results['Reduced Chi^2'] < 1.5] for key in band_keys],label=[b for b in band_list])
#         ax[count].set_title('lens light \n {} \n {}'.format(prof,param))
#         ax[count].set_xlabel('Value')
#         ax[count].set_ylabel('# Occurences')
#         ax[count].legend()
#         count += 1
            
                
            
            

# R_source = deepcopy(dict_results['source']['r Band: SERSIC_ELLIPSE']['R_sersic'][dict_results['Reduced Chi^2'] < 1.5])
# ax[count].hist(R_source[R_source <= 2])
# ax[count].set_title('r Band: Source \n SERSIC_ELLIPSE \n R_sersic')
# ax[count].set_xlabel('Value')
# ax[count].set_ylabel('# Occurences')
# count += 1
for i,a in enumerate(ax):
    if i >= count:
        a.set_axis_off()
f.savefig(results_path_rings + '/histograms_compare_DES_CFIS',dpi = 500)
plt.close(f)




    
    
    








