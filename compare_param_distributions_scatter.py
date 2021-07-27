import sys
if sys.platform[:5] == 'linux':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import pandas as pd
from pylab import figure, cm
from matplotlib.colors import LogNorm
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Util.param_util import ellipticity2phi_q
from lenstronomy.Util.param_util import shear_cartesian2polar
from os import walk
from os import listdir
from os.path import isfile, join, exists
from Lens_Modeling_Auto.auto_modeling_functions import df_2_dict


#### Create dataframe from csv file(s) ####
deltaPix = 0.27
obj_name_location = 0
band_list = ['g','r','i']

lens_model_list = ['SIE','SHEAR']
# lens_model_list = ['PEMD','SHEAR']
source_model_list = ['SERSIC_ELLIPSE']
lens_light_model_list = ['SERSIC_ELLIPSE']


#### Folder for Results ####
results_path = '<destination folder for plot results>'
if not exists(results_path):
    os.mkdir(results_path)

    
#### First set of modeling results ####
path_1 = '<parent folder>/'
csv_paths_1 = path_1 + '<modeling results folder>/'

df_final_1 = pd.read_csv(csv_paths_1 + 'full_results_sorted.csv',delimiter =',')

# for i in range(len(csv_paths_1)-1):
#     df = pd.read_csv(csv_paths_1[i+1] + 'full_results.csv',delimiter =',')
#     df_list_1.append(df.loc[1:,:])    
    
# df_final_1 = pd.concat(df_list_1,axis=0,ignore_index=True) #Final dataframe of data set 1

data_dict_1 = df_2_dict(df_final_1,band_list,lens_model_list,source_model_list,lens_light_model_list) #data set 1 as dictionary

### Second set of modeling results ####
path_2 = '<parent folder>/'
csv_paths_2 = path_2 + '<modeling results folder>/'

df_final_2 = pd.read_csv(csv_paths_2 + 'full_results_sorted.csv',delimiter =',')

# for i in range(len(csv_paths_2)-1):
#     df = pd.read_csv(csv_paths_2[i+1] + 'full_results.csv',delimiter =',')
#     df_list_2.append(df.loc[1:,:])    
    
# df_final_2 = pd.concat(df_list_2,axis=0,ignore_index=True)

data_dict_2 = df_2_dict(df_final_2,band_list,lens_model_list,source_model_list,lens_light_model_list)

#print(data_dict['shear']['gamma'])


### Third set of modeling results ####
deltaPix_3 = 0.1857
obj_name_location_3 = 1
band_list_3 = ['r']
### Third set of modeling results ####
path_3 = '<parent folder>/'
csv_paths_3 = path_3 + '<modeling results folder>/'

df_final_3 = pd.read_csv(csv_paths_3 + 'full_results_sorted.csv',delimiter =',')

# for i in range(len(csv_paths_3)-1):
#     df = pd.read_csv(csv_paths_3[i+1] + 'full_results.csv',delimiter =',')
#     df_list_3.append(df.loc[1:,:])    
    
# df_final_3 = pd.concat(df_list_3,axis=0,ignore_index=True)

data_dict_3 = df_2_dict(df_final_3,band_list_3,lens_model_list,source_model_list,lens_light_model_list)

cut = 3.0

f,ax = plt.subplots(6,5,figsize=(20,20))
f.subplots_adjust(hspace=0.75,wspace=0.5)
ax = ax.ravel()

print(type(data_dict_1['Reduced Chi^2']))
print(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < 5.])
ax[0].hist([np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < 5.],np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < 5.],np.array(data_dict_3['Reduced Chi^2'])[np.array(data_dict_3['Reduced Chi^2']) < 5.]],label=['DES lenses', 'Ring Galaxies','CFIS lenses'],density=True)
ax[0].set_title('Reduced Chi^2')
ax[0].legend()

# ax[1].hist([np.array(data_dict_1['shear']['gamma'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_2['shear']['gamma'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_3['shear']['gamma'])[np.array(data_dict_3['Reduced Chi^2']) < cut]],label=['DES lenses', 'Ring Galaxies','CFIS lenses'])
# ax[1].set_title('Shear Strengths')
# ax[1].legend()

# ax[2].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['shear']['gamma'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[2].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['shear']['gamma'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[2].scatter(np.array(data_dict_3['Reduced Chi^2'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['shear']['gamma'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
# ax[2].set_title('Shear vs Chi^2')
# ax[2].legend()

xl_1,yl_1 = data_dict_1['lens']['SIE']['center_x'],data_dict_1['lens']['SIE']['center_y']
xs_1,ys_1 = data_dict_1['source']['r Band: SERSIC_ELLIPSE']['center_x'],data_dict_1['source']['r Band: SERSIC_ELLIPSE']['center_y']
data_dict_1['lens_source_disp'] = np.sqrt( (xl_1 - xs_1)**2  + (yl_1 - ys_1)**2 )

xl_2,yl_2 = data_dict_2['lens']['SIE']['center_x'],data_dict_2['lens']['SIE']['center_y']
xs_2,ys_2 = data_dict_2['source']['r Band: SERSIC_ELLIPSE']['center_x'],data_dict_2['source']['r Band: SERSIC_ELLIPSE']['center_y']
data_dict_2['lens_source_disp'] = np.sqrt( (xl_2 - xs_2)**2  + (yl_2 - ys_2)**2 )

xl_3,yl_3 = data_dict_3['lens']['SIE']['center_x'],data_dict_3['lens']['SIE']['center_y']
xs_3,ys_3 = data_dict_3['source']['r Band: SERSIC_ELLIPSE']['center_x'],data_dict_3['source']['r Band: SERSIC_ELLIPSE']['center_y']
data_dict_3['lens_source_disp'] = np.sqrt( (xl_3 - xs_3)**2  + (yl_3 - ys_3)**2 )

ax[1].hist([np.array(data_dict_1['lens_source_disp'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_source_disp'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_3['lens_source_disp'])[np.array(data_dict_3['Reduced Chi^2']) < cut]],label=['DES lenses', 'Ring Galaxies','CFIS lenses'],density=True)
ax[1].set_title('Lens-Source Alignment')
ax[1].legend()

ax[2].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_source_disp'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
ax[2].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_source_disp'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
ax[2].scatter(np.array(data_dict_3['Reduced Chi^2'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['lens_source_disp'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
ax[2].set_title('Lens-Source Alignment vs Chi^2')
ax[2].legend()
ax[2].set_ylim(0,0.75)

ax[3].hist([np.array(data_dict_1['lens']['SIE']['theta_E'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_2['lens']['SIE']['theta_E'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_3['lens']['SIE']['theta_E'])[np.array(data_dict_3['Reduced Chi^2']) < cut]],label=['DES lenses', 'Ring Galaxies','CFIS lenses'],density=True)
ax[3].set_title('theta_E')
ax[3].legend(prop={'size': 6})

# ax[3].hist([np.array(data_dict_1['lens']['SIE']['theta_E'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_3['lens']['SIE']['theta_E'])[np.array(data_dict_3['Reduced Chi^2']) < cut]],label=['DES lenses','CFIS lenses'])
# ax[3].set_title('theta_E')
# ax[3].legend(prop={'size': 6})

ax[4].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens']['SIE']['q'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
ax[4].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens']['SIE']['q'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
ax[4].scatter(np.array(data_dict_3['Reduced Chi^2'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['lens']['SIE']['q'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
ax[4].set_title('Lens Aspect Ratio (b/a)')
ax[4].legend()

ax[5].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens']['SIE']['theta_E'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
ax[5].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens']['SIE']['theta_E'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
ax[5].scatter(np.array(data_dict_3['Reduced Chi^2'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['lens']['SIE']['theta_E'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
ax[5].set_title('theta_E vs Chi^2')
ax[5].set_xlabel('Reduced Chi^2')
ax[5].set_ylabel('theta_E (Arcsec)')
ax[5].legend(prop={'size': 6})

# ax[5].hist([data_dict_1['source']['g']['R_sersic'],data_dict_2['source']['g']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
# ax[5].set_title('g band: \n R_sersic (source)')
# ax[5].legend()

# ax[6].hist([data_dict_1['source']['r']['R_sersic'],data_dict_2['source']['r']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
# ax[6].set_title('r band: \n R_sersic (source)')
# ax[6].legend()

# ax[7].hist([data_dict_1['source']['i']['R_sersic'],data_dict_2['source']['i']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
# ax[7].set_title('i band: \n R_sersic (source)')
# ax[7].legend()

# ax[8].hist([data_dict_1['lens_light']['g']['R_sersic'],data_dict_2['lens_light']['g']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
# ax[8].set_title('g band: \n R_sersic (lens_light)')
# ax[8].legend()

# ax[9].hist([data_dict_1['lens_light']['r']['R_sersic'],data_dict_2['lens_light']['r']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
# ax[9].set_title('r band: \n R_sersic (lens_light)')
# ax[9].legend()

# ax[10].hist([data_dict_1['lens_light']['i']['R_sersic'],data_dict_2['lens_light']['i']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
# ax[10].set_title('i band: \n R_sersic (lens_light)')
# ax[10].legend()

# ax[11].hist([data_dict_1['source']['g']['n_sersic'],data_dict_2['source']['g']['n_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
# ax[11].set_title('g band: \n n_sersic (source)')
# ax[11].legend()

# ax[12].hist([data_dict_1['source']['r']['n_sersic'],data_dict_2['source']['r']['n_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
# ax[12].set_title('r band: \n n_sersic (source)')
# ax[12].legend()

# ax[13].hist([data_dict_1['source']['i']['n_sersic'],data_dict_2['source']['i']['n_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
# ax[13].set_title('i band: \n n_sersic (source)')
# ax[13].legend()

# ax[6].scatter(np.array(data_dict_1['lens_light']['g']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['g']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[6].scatter(np.array(data_dict_2['lens_light']['g']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['g']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[6].set_title('g band: \n R (source) vs R (lens_light)')
# ax[6].legend()

ax[7].scatter(np.array(data_dict_1['lens_light']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
ax[7].scatter(np.array(data_dict_2['lens_light']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
ax[7].scatter(np.array(data_dict_3['lens_light']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['source']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
# ax[7].set_title('r band: \n R (source) vs R (lens_light)')
ax[7].set_xlabel('R_sersic (lens light)')
ax[7].set_ylabel('R_sersic (source)')
ax[7].legend()

# ax[8].scatter(np.array(data_dict_1['lens_light']['i']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['i']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[8].scatter(np.array(data_dict_2['lens_light']['i']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['i']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[8].set_title('i band: \n R (source) vs R (lens_light)')
# ax[8].legend()

# ax[17].hist([data_dict_1['source']['g']['q'],data_dict_2['source']['g']['q']],label=['Lens Candidates', 'Ring Galaxies'])
# ax[17].set_title('g band: \n Aspect Ratio (b/a) (source)')
# ax[17].legend()

# ax[9].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['g']['q'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[9].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['g']['q'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[9].set_title('g band: \n Aspect Ratio (b/a) (lens_light) vs Chi^2')
# ax[9].legend()

ax[10].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['r Band: SERSIC_ELLIPSE']['q'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
ax[10].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['r Band: SERSIC_ELLIPSE']['q'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
ax[10].scatter(np.array(data_dict_3['Reduced Chi^2'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['lens_light']['r Band: SERSIC_ELLIPSE']['q'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
ax[10].set_title('r band: \n Aspect Ratio (b/a) (lens_light) vs Chi^2')
ax[10].legend()

# ax[11].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['i']['q'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[11].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['i']['q'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[11].set_title('i band: \n Aspect Ratio (b/a) (lens_light) vs Chi^2')
# ax[11].legend()


# ax[12].scatter(np.array(data_dict_1['lens_light']['g']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['g']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[12].scatter(np.array(data_dict_2['lens_light']['g']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['g']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[12].set_title('g band: \n R vs n (lens_light)')
# ax[12].legend()

ax[13].scatter(np.array(data_dict_1['lens_light']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
ax[13].scatter(np.array(data_dict_2['lens_light']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
ax[13].scatter(np.array(data_dict_3['lens_light']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['lens_light']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
ax[13].set_title('r band: \n R vs n (lens_light)')
ax[13].legend()


# ax[14].scatter(np.array(data_dict_1['lens_light']['i']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['i']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[14].scatter(np.array(data_dict_2['lens_light']['i']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['i']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[14].set_title('i band: \n R vs n (lens_light)')
# ax[14].legend()

# ax[23].scatter(data_dict_1['lens_light']['g']['phi'],data_dict_1['lens']['phi'],label='lens Candidates')
# ax[23].scatter(data_dict_2['lens_light']['g']['phi'],data_dict_2['lens']['phi'],label='Ring Galaxies')
# ax[23].set_title('g band: \n phi(lens mass) vs phi(lens light)')
# ax[23].legend()

# ax[24].scatter(data_dict_1['lens_light']['r']['phi'],data_dict_1['lens']['phi'],label='lens Candidates')
# ax[24].scatter(data_dict_2['lens_light']['r']['phi'],data_dict_2['lens']['phi'],label='Ring Galaxies')
# ax[24].set_title('r band: \n phi(lens mass) vs phi(lens light)')
# ax[24].legend()

# ax[25].scatter(data_dict_1['lens_light']['i']['phi'],data_dict_1['lens']['phi'],label='lens Candidates')
# ax[25].scatter(data_dict_2['lens_light']['i']['phi'],data_dict_2['lens']['phi'],label='Ring Galaxies')
# ax[25].set_title('i band: \n phi(lens mass) vs phi(lens light)')
# ax[25].legend()

# ax[15].scatter(np.array(data_dict_1['source']['g']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['g']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[15].scatter(np.array(data_dict_2['source']['g']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['g']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[15].set_title('g band: \n R vs n (source)')
# ax[15].legend()

ax[16].scatter(np.array(data_dict_1['source']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
ax[16].scatter(np.array(data_dict_2['source']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
ax[16].scatter(np.array(data_dict_3['source']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['source']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
ax[16].set_title('r band: \n R vs n (source)')
ax[16].legend()


# ax[17].scatter(np.array(data_dict_1['source']['i']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['i']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[17].scatter(np.array(data_dict_2['source']['i']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['i']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[17].set_title('i band: \n R vs n (source)')
# ax[17].legend()

# ax[18].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['g']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[18].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['g']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[18].set_title('g band: \n R (lens light) vs Chi^2')
# ax[18].legend()

ax[19].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
ax[19].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
ax[19].scatter(np.array(data_dict_3['Reduced Chi^2'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['lens_light']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
ax[19].set_title('r band: \n R (lens light) vs Chi^2')
ax[19].legend()

# ax[20].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['i']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[20].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['i']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[20].set_title('i band: \n R (lens light) vs Chi^2')
# ax[20].legend()


# ax[21].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['g']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[21].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['g']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[21].set_title('g band: \n R (source) vs Chi^2')
# ax[21].legend()

ax[22].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
ax[22].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
ax[22].scatter(np.array(data_dict_3['Reduced Chi^2'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['source']['r Band: SERSIC_ELLIPSE']['R_sersic'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
ax[22].set_title('r band: \n R (source) vs Chi^2')
ax[22].legend()

# ax[23].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['i']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[23].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['i']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[23].set_title('i band: \n R (source) vs Chi^2')
# ax[23].legend()


# ax[24].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['g']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[24].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['g']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[24].set_title('g band: \n n (lens light) vs Chi^2')
# ax[24].legend()

ax[25].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
ax[25].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
ax[25].scatter(np.array(data_dict_3['Reduced Chi^2'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['lens_light']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
ax[25].set_title('r band: \n n (lens light) vs Chi^2')
ax[25].legend()

# ax[26].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['lens_light']['i']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[26].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['lens_light']['i']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[26].set_title('i band: \n n (lens light) vs Chi^2')
# ax[26].legend()



# ax[27].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['g']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[27].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['g']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[27].set_title('g band: \n n (source) vs Chi^2')
# ax[27].legend()

ax[28].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
ax[28].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
ax[28].scatter(np.array(data_dict_3['Reduced Chi^2'])[np.array(data_dict_3['Reduced Chi^2']) < cut],np.array(data_dict_3['source']['r Band: SERSIC_ELLIPSE']['n_sersic'])[np.array(data_dict_3['Reduced Chi^2']) < cut],label='CFIS lenses')
# ax[28].set_title('r band: \n n (source) vs Chi^2')
ax[28].set_xlabel('reduced Chi^2')
ax[28].set_ylabel('n_sersic (source)')
ax[28].legend()

# ax[29].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < cut],np.array(data_dict_1['source']['i']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < cut],label='DES lenses')
# ax[29].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < cut],np.array(data_dict_2['source']['i']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < cut],label='Ring Galaxies')
# ax[29].set_title('i band: \n n (source) vs Chi^2')
# ax[29].legend()



# plt.subplot(5,5,2)
f.tight_layout()
f.savefig(results_path + '/compare.png',dpi = 200)
plt.close(f)
# plt.show()




    
    
    








