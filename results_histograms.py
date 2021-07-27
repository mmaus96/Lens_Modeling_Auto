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

# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_lenses/Sure_Lens/'
# csv_path = path + 'SIE_lens/results_May31/'
# results_path = path + 'SIE_lens/results_May31'

path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Group1/'
csv_path = path + 'SIE_lens/results_Jun1/'
results_path = path + 'SIE_lens/results_Jun1'

# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/ringcatalog/'
# csv_path = path + 'results_May3/'
# results_path = path + 'results_May3'

if not exists(results_path):
    os.mkdir(results_path)
    
lens_model_list = ['SIE','SHEAR']
# lens_model_list = ['PEMD','SHEAR']
source_model_list = ['SERSIC_ELLIPSE']
lens_light_model_list = ['SERSIC_ELLIPSE']
band_list = ['g','r','i']
# band_list = ['r']

df = pd.read_csv(csv_path + 'full_results_sorted.csv',delimiter =',')

add_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_lenses/Sure_Lens/SIE_lens/new_lenses/results_new_lenses_3/'

df_add = pd.read_csv(add_path + 'full_results_sorted.csv',delimiter =',')

# df_final = pd.concat([df_final, df_add])
df_final = df.append(df_add.loc[1].transpose())
df_final=df_final.reset_index(drop=True)

dict_results = df_2_dict(df_final,band_list,lens_model_list,source_model_list,lens_light_model_list)

# print(data_dict['shear']['gamma'])


num_plots = 1 
cols = 9

for prof in lens_model_list:
    results = np.array(list(dict_results['lens'][prof].items()))
    print(len(results))
    num_plots += (len(results) - 2)
    
# for band in band_list:
#     for prof in source_model_list:
#         key = '{} Band: {}'.format(band,prof)
#         results = np.array(list(dict_results['source'][key].items()))
#         print(len(results))
#         num_plots += (len(results) - 2)

#     for prof in lens_light_model_list:
#         key = '{} Band: {}'.format(band,prof)
#         results = np.array(list(dict_results['lens_light'][key].items()))
#         print(len(results))
#         num_plots += (len(results) - 2)


for prof in source_model_list:
    key = '{} Band: {}'.format(band_list[0],prof)
    results = np.array(list(dict_results['source'][key].items()))
    print(len(results))
    num_plots += (len(results) - 2)

for prof in lens_light_model_list:
    key = '{} Band: {}'.format(band_list[0],prof)
    results = np.array(list(dict_results['lens_light'][key].items()))
    print(len(results))
    num_plots += (len(results) - 2)
        
print(num_plots)

rows = ceil(num_plots /  cols)

f,axes = plt.subplots(rows,cols, figsize = (cols*5,rows*5))
f.subplots_adjust(hspace=0.5,wspace=0.5)
count = 0
ax = axes.ravel()

ax[0].hist(dict_results['Reduced Chi^2'][dict_results['Reduced Chi^2'] < 5])
ax[0].set_title('Reduced Chi^2')
ax[0].set_xlabel('Value')
ax[0].set_ylabel('# Occurences')

count +=1

for prof in lens_model_list:
    results = np.array(list(dict_results['lens'][prof].items()))
    for i in range(len(results)):
        if (results[i][0] == 'ra_0') or (results[i][0] == 'dec_0') or (results[i][0] == 'center_x') or (results[i][0] == 'center_y'):
            continue
        ax[count].hist(results[i][1][dict_results['Reduced Chi^2'] < 1.5])
        ax[count].set_title('Lens \n {} \n {}'.format(prof,results[i][0]))
        ax[count].set_xlabel('Value')
        ax[count].set_ylabel('# Occurences')
        count += 1
    
# for band in band_list:
#     for prof in source_model_list:
#         key = '{} Band: {}'.format(band,prof)
#         results = np.array(list(dict_results['source'][key].items()))
#         for i in range(len(results)):
#             if (results[i][0] == 'center_x') or (results[i][0] == 'center_y'):
#                 continue
#             ax[count].hist(results[i][1][dict_results['Reduced Chi^2'] < 1.5])
#             ax[count].set_title('{} Band: Source \n {} \n {}'.format(band,prof,results[i][0]))
#             ax[count].set_xlabel('Value')
#             ax[count].set_ylabel('# Occurences')
#             count += 1

#     for prof in lens_light_model_list:
#         key = '{} Band: {}'.format(band,prof)
#         results = np.array(list(dict_results['lens_light'][key].items()))
#         for i in range(len(results)):
#             if (results[i][0] == 'center_x') or (results[i][0] == 'center_y'):
#                 continue
#             ax[count].hist(results[i][1][dict_results['Reduced Chi^2'] < 1.5])
#             ax[count].set_title('{} Band: Lens Light \n {} \n {}'.format(band,prof,results[i][0]))
#             ax[count].set_xlabel('Value')
#             ax[count].set_ylabel('# Occurences')
#             count += 1

for prof in source_model_list:
    key0 = '{} Band: {}'.format(band_list[0],prof)
    source_params = np.array(list(dict_results['source'][key].items()))[:,0]
    for i,param in enumerate(source_params):
        if (param == 'center_x') or (param == 'center_y'):
            continue

        band_keys = []
        for band in band_list:
            key = '{} Band: {}'.format(band,prof)
            band_keys.append(key)
#             results = np.array(list(dict_results['source'][key].items()))
        ax[count].hist([dict_results['source'][key][param][dict_results['Reduced Chi^2'] < 1.5] for key in band_keys],label=[b for b in band_list])
        ax[count].set_title('Source \n {} \n {}'.format(prof,param))
        ax[count].set_xlabel('Value')
        ax[count].set_ylabel('# Occurences')
        ax[count].legend()
        count += 1

for prof in lens_light_model_list:
    key0 = '{} Band: {}'.format(band_list[0],prof)
    lens_light_params = np.array(list(dict_results['lens_light'][key].items()))[:,0]
    for i,param in enumerate(lens_light_params):
        if (param == 'center_x') or (param == 'center_y'):
            continue

        band_keys = []
        for band in band_list:
            key = '{} Band: {}'.format(band,prof)
            band_keys.append(key)
#             results = np.array(list(dict_results['source'][key].items()))
        ax[count].hist([dict_results['lens_light'][key][param][dict_results['Reduced Chi^2'] < 1.5] for key in band_keys],label=[b for b in band_list])
        ax[count].set_title('lens light \n {} \n {}'.format(prof,param))
        ax[count].set_xlabel('Value')
        ax[count].set_ylabel('# Occurences')
        ax[count].legend()
        count += 1
            
                
            
            

# R_source = deepcopy(dict_results['source']['r Band: SERSIC_ELLIPSE']['R_sersic'][dict_results['Reduced Chi^2'] < 1.5])
# ax[count].hist(R_source[R_source <= 2])
# ax[count].set_title('r Band: Source \n SERSIC_ELLIPSE \n R_sersic')
# ax[count].set_xlabel('Value')
# ax[count].set_ylabel('# Occurences')
# count += 1
for i,a in enumerate(ax):
    if i >= count:
        a.set_axis_off()
f.savefig(results_path + '/histograms_combine_bands',dpi = 500)
plt.close(f)

#custom figs

fig,axes = plt.subplots(3,2, figsize = (10,15))
ax = axes.ravel()
fontsize = 20
cut = 1.5

count = 0
prof = 'SIE'
param = 'theta_E'
ax[count].hist([dict_results['lens'][prof][param][dict_results['Reduced Chi^2'] <= cut],
               dict_results['lens'][prof][param][dict_results['Reduced Chi^2'] > cut]],
              label = [r'$\chi^2 \leq {}$'.format(cut),r'$\chi^2 > {}$'.format(cut)], color = ['green','red'],bins=20,histtype='step',lw=2)
# ax[count].hist([dict_results['lens'][prof][param],dict_results['lens'][prof][param][dict_results['Reduced Chi^2'] <= cut],
#                dict_results['lens'][prof][param][dict_results['Reduced Chi^2'] > cut]],
#               label = ['All results',r'$\chi^2 \leq {}$'.format(cut),r'$\chi^2 > {}$'.format(cut)], 
#                color = ['blue','green','red'])
ax[count].set_title('Einstein Radius ($R_E$)',fontsize=fontsize)
ax[count].set_xlabel(r'$R_E$ (arcsec)',fontsize=fontsize)
ax[count].set_ylabel('# Occurences',fontsize=fontsize)
ax[count].tick_params(axis='x', labelsize=fontsize)
ax[count].tick_params(axis='y', labelsize=fontsize)
ax[count].legend(fontsize=fontsize)
count += 1

prof = 'SHEAR'
param = 'gamma'
ax[count].hist([dict_results['lens'][prof][param][dict_results['Reduced Chi^2'] <= cut],
               dict_results['lens'][prof][param][dict_results['Reduced Chi^2'] > cut]],
              label = [r'$\chi^2 \leq {}$'.format(cut),r'$\chi^2 > {}$'.format(cut)], color = ['green','red'],bins=20,histtype='step',lw=2)
# ax[count].hist([dict_results['lens'][prof][param],dict_results['lens'][prof][param][dict_results['Reduced Chi^2'] <= cut],
#                dict_results['lens'][prof][param][dict_results['Reduced Chi^2'] > cut]],
#               label = ['All results',r'$\chi^2 \leq {}$'.format(cut),r'$\chi^2 > {}$'.format(cut)], color = ['blue','green','red'])
ax[count].set_title('Shear Strength ($\gamma_{ext}$)', fontsize=fontsize)
ax[count].set_xlabel(r'$\gamma_{ext}$',fontsize=fontsize)
ax[count].set_ylabel('# Occurences',fontsize=fontsize)
ax[count].tick_params(axis='x', labelsize=fontsize)
ax[count].tick_params(axis='y', labelsize=fontsize)
ax[count].legend(fontsize=fontsize)
count += 1

prof = 'SERSIC_ELLIPSE'

names = [r'$R_{eff}$ (arcsec)',r'$n_s$']
for i,param in enumerate(['R_sersic','n_sersic']):
    band = 'r'
    key = '{} Band: {}'.format(band,prof)
    ax[count].hist([dict_results['lens_light'][key][param][dict_results['Reduced Chi^2'] <= cut],
                   dict_results['lens_light'][key][param][dict_results['Reduced Chi^2'] > cut]],
                   label = [r'$\chi^2 \leq {}$'.format(cut),r'$\chi^2 > {}$'.format(cut)], color = ['green','red'],bins=20,histtype='step',lw=2)
#     ax[count].hist([dict_results['lens_light'][key][param],
#                     dict_results['lens_light'][key][param][dict_results['Reduced Chi^2'] <= cut],
#                    dict_results['lens_light'][key][param][dict_results['Reduced Chi^2'] > cut]],
#                    label = ['All results',r'$\chi^2 \leq {}$'.format(cut),r'$\chi^2 > {}$'.format(cut)], 
#                    color = ['blue','green','red'])
    ax[count].set_title('Lens Light: {}'.format(names[i]), fontsize=fontsize)
    ax[count].set_xlabel('{}'.format(names[i]),fontsize=fontsize)
    ax[count].set_ylabel('# Occurences',fontsize=fontsize)
    ax[count].tick_params(axis='x', labelsize=fontsize)
    ax[count].tick_params(axis='y', labelsize=fontsize)
    ax[count].legend(fontsize=fontsize)
    count += 1  


# for param in ['R_sersic','n_sersic']:
#     band_keys = []
#     for band in band_list:
#         key = '{} Band: {}'.format(band,prof)
#         band_keys.append(key)
#     ax[count].hist([dict_results['source'][key][param][dict_results['Reduced Chi^2'] < 1.5] for key in band_keys],label=[b for b in band_list])
#     ax[count].set_title('Source Light: {} \n {}'.format(prof,param))
#     ax[count].set_xlabel('Value')
#     ax[count].set_ylabel('# Occurences')
#     # ax[count].legend()
#     count += 1

for i,param in enumerate(['R_sersic','n_sersic']):
    band = 'r'
    key = '{} Band: {}'.format(band,prof)
    ax[count].hist([dict_results['source'][key][param][dict_results['Reduced Chi^2'] <= cut],
                   dict_results['source'][key][param][dict_results['Reduced Chi^2'] > cut]],
                   label = [r'$\chi^2 \leq {}$'.format(cut),r'$\chi^2 > {}$'.format(cut)], color = ['green','red'],bins=20,histtype='step',lw=2)
#     ax[count].hist([dict_results['source'][key][param],
#                     dict_results['source'][key][param][dict_results['Reduced Chi^2'] <= cut],
#                    dict_results['source'][key][param][dict_results['Reduced Chi^2'] > cut]],
#                    label = ['All results',r'$\chi^2 \leq {}$'.format(cut),r'$\chi^2 > {}$'.format(cut)], color = ['blue','green','red'])
    ax[count].set_title('Source Light: {}'.format(names[i]), fontsize=fontsize)
    ax[count].set_xlabel('{}'.format(names[i]),fontsize=fontsize)
    ax[count].set_ylabel('# Occurences',fontsize=fontsize)
    ax[count].tick_params(axis='x', labelsize=fontsize)
    ax[count].tick_params(axis='y', labelsize=fontsize)
    ax[count].legend(fontsize=fontsize)
    count += 1


# for param in ['R_sersic','n_sersic']:
#     band_keys = []
#     for band in band_list:
#         key = '{} Band: {}'.format(band,prof)
#         band_keys.append(key)
#     ax[count].hist([dict_results['lens_light'][key][param][dict_results['Reduced Chi^2'] < 1.5] for key in band_keys],label=[b for b in band_list])
#     ax[count].set_title('Lens Light: {} \n {}'.format(prof,param))
#     ax[count].set_xlabel('Value')
#     ax[count].set_ylabel('# Occurences')
#     # ax[count].legend()
#     count += 1    
    

  
    
fig.subplots_adjust(hspace=None,wspace=0.5)
fig.tight_layout()
fig.savefig(results_path + '/histograms_custom_bins20.pdf',dpi = 100)
plt.close(fig)








