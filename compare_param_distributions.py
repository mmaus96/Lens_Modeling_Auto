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


#### Folder for Results ####
results_path = '<destination folder for plot results>'
if not exists(results_path):
    os.mkdir(results_path)

    
#### First set of modeling results ####
path_1 = '<parent folder>/'
csv_paths_1 = [path_1 + '<modeling results folder>/'] #can combine multiple data sets

df_list_1 = [pd.read_csv(csv_paths_1[0] + 'full_results.csv',delimiter =',')]

for i in range(len(csv_paths_1)-1):
    df = pd.read_csv(csv_paths_1[i+1] + 'full_results.csv',delimiter =',')
    df_list_1.append(df.loc[:,:])    
    
df_final_1 = pd.concat(df_list_1,axis=0,ignore_index=True) #Final dataframe of data set 1

data_dict_1 = df_2_dict(df_final_1,band_list,obj_name_location) #data set 1 as dictionary

### Second set of modeling results ####
path_2 = '<parent folder>/'
csv_paths_2 = [path_2 + '<modeling results folder>/'] #can combine multiple data sets

df_list_2 = [pd.read_csv(csv_paths_2[0] + 'full_results.csv',delimiter =',')]

for i in range(len(csv_paths_2)-1):
    df = pd.read_csv(csv_paths_2[i+1] + 'full_results.csv',delimiter =',')
    df_list_2.append(df.loc[:,:])    
    
df_final_2 = pd.concat(df_list_2,axis=0,ignore_index=True)

data_dict_2 = df_2_dict(df_final_2,band_list,obj_name_location)

#print(data_dict['shear']['gamma'])



f,ax = plt.subplots(6,5,figsize=(20,20))
f.subplots_adjust(hspace=0.75,wspace=0.5)
ax = ax.ravel()

print(type(data_dict_1['Reduced Chi^2']))
print(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < 5.])
ax[0].hist([np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < 5.],np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < 5.]],label=['Lens Candidates', 'Ring Galaxies'])
ax[0].set_title('Reduced Chi^2')
ax[0].legend()

ax[1].hist([data_dict_1['shear']['gamma'],data_dict_2['shear']['gamma']],label=['Lens Candidates', 'Ring Galaxies'])
ax[1].set_title('Shear Strengths')
ax[1].legend()

ax[2].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < 5.],np.array(data_dict_1['shear']['gamma'])[np.array(data_dict_1['Reduced Chi^2']) < 5.],label='Lens Candidates')
ax[2].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < 5.],np.array(data_dict_2['shear']['gamma'])[np.array(data_dict_2['Reduced Chi^2']) < 5.],label='Ring Galaxies')
ax[2].set_title('Shear vs Chi^2')
ax[2].legend()

ax[3].hist([data_dict_1['lens']['theta_E'],data_dict_2['lens']['theta_E']],label=['Lens Candidates', 'Ring Galaxies'])
ax[3].set_title('theta_E')
ax[3].legend()

ax[4].hist([data_dict_1['lens']['q'],data_dict_2['lens']['q']],label=['Lens Candidates', 'Ring Galaxies'])
ax[4].set_title('Lens Aspect Ratio (b/a)')
ax[4].legend()

ax[5].hist([data_dict_1['source']['g']['R_sersic'],data_dict_2['source']['g']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[5].set_title('g band: \n R_sersic (source)')
ax[5].legend()

ax[6].hist([data_dict_1['source']['r']['R_sersic'],data_dict_2['source']['r']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[6].set_title('r band: \n R_sersic (source)')
ax[6].legend()

ax[7].hist([data_dict_1['source']['i']['R_sersic'],data_dict_2['source']['i']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[7].set_title('i band: \n R_sersic (source)')
ax[7].legend()

ax[8].hist([data_dict_1['lens_light']['g']['R_sersic'],data_dict_2['lens_light']['g']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[8].set_title('g band: \n R_sersic (lens_light)')
ax[8].legend()

ax[9].hist([data_dict_1['lens_light']['r']['R_sersic'],data_dict_2['lens_light']['r']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[9].set_title('r band: \n R_sersic (lens_light)')
ax[9].legend()

ax[10].hist([data_dict_1['lens_light']['i']['R_sersic'],data_dict_2['lens_light']['i']['R_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[10].set_title('i band: \n R_sersic (lens_light)')
ax[10].legend()

ax[11].hist([data_dict_1['source']['g']['n_sersic'],data_dict_2['source']['g']['n_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[11].set_title('g band: \n n_sersic (source)')
ax[11].legend()

ax[12].hist([data_dict_1['source']['r']['n_sersic'],data_dict_2['source']['r']['n_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[12].set_title('r band: \n n_sersic (source)')
ax[12].legend()

ax[13].hist([data_dict_1['source']['i']['n_sersic'],data_dict_2['source']['i']['n_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[13].set_title('i band: \n n_sersic (source)')
ax[13].legend()

ax[14].hist([data_dict_1['lens_light']['g']['n_sersic'],data_dict_2['lens_light']['g']['n_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[14].set_title('g band: \n n_sersic (lens_light)')
ax[14].legend()

ax[15].hist([data_dict_1['lens_light']['r']['n_sersic'],data_dict_2['lens_light']['r']['n_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[15].set_title('r band: \n n_sersic (lens_light)')
ax[15].legend()

ax[16].hist([data_dict_1['lens_light']['i']['n_sersic'],data_dict_2['lens_light']['i']['n_sersic']],label=['Lens Candidates', 'Ring Galaxies'])
ax[16].set_title('i band: \n n_sersic (lens_light)')
ax[16].legend()

ax[17].hist([data_dict_1['source']['g']['q'],data_dict_2['source']['g']['q']],label=['Lens Candidates', 'Ring Galaxies'])
ax[17].set_title('g band: \n Aspect Ratio (b/a) (source)')
ax[17].legend()

ax[18].hist([data_dict_1['source']['r']['q'],data_dict_2['source']['r']['q']],label=['Lens Candidates', 'Ring Galaxies'])
ax[18].set_title('r band: \n Aspect Ratio (b/a) (source)')
ax[18].legend()

ax[19].hist([data_dict_1['source']['i']['q'],data_dict_2['source']['i']['q']],label=['Lens Candidates', 'Ring Galaxies'])
ax[19].set_title('i band: \n Aspect Ratio (b/a) (source)')
ax[19].legend()

ax[20].hist([data_dict_1['lens_light']['g']['q'],data_dict_2['lens_light']['g']['q']],label=['Lens Candidates', 'Ring Galaxies'])
ax[20].set_title('g band: \n Aspect Ratio (b/a) (lens_light)')
ax[20].legend()

ax[21].hist([data_dict_1['lens_light']['r']['q'],data_dict_2['lens_light']['r']['q']],label=['Lens Candidates', 'Ring Galaxies'])
ax[21].set_title('r band: \n Aspect Ratio (b/a) (lens_light)')
ax[21].legend()

ax[22].hist([data_dict_1['lens_light']['i']['q'],data_dict_2['lens_light']['i']['q']],label=['Lens Candidates', 'Ring Galaxies'])
ax[22].set_title('i band: \n Aspect Ratio (b/a) (lens_light)')
ax[22].legend()

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

ax[23].scatter(data_dict_1['source']['g']['n_sersic'],data_dict_1['source']['g']['R_sersic'],label='lens Candidates')
ax[23].scatter(data_dict_2['source']['g']['n_sersic'],data_dict_2['source']['g']['R_sersic'],label='Ring Galaxies')
ax[23].set_title('g band: \n R vs n (source)')
ax[23].legend()

ax[24].scatter(data_dict_1['source']['r']['n_sersic'],data_dict_1['source']['r']['R_sersic'],label='lens Candidates')
ax[24].scatter(data_dict_2['source']['r']['n_sersic'],data_dict_2['source']['r']['R_sersic'],label='Ring Galaxies')
ax[24].set_title('r band: \n R vs n (source)')
ax[24].legend()

ax[25].scatter(data_dict_1['source']['i']['n_sersic'],data_dict_1['source']['i']['R_sersic'],label='lens Candidates')
ax[25].scatter(data_dict_2['source']['i']['n_sersic'],data_dict_2['source']['i']['R_sersic'],label='Ring Galaxies')
ax[25].set_title('i band: \n R vs n (source)')
ax[25].legend()


ax[26].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < 5.],np.array(data_dict_1['source']['g']['R_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < 5.],label='Lens Candidates')
ax[26].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < 5.],np.array(data_dict_2['source']['g']['R_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < 5.],label='Ring Galaxies')
ax[26].set_title('g band: \n R (source) vs Chi^2')
ax[26].legend()


ax[27].scatter(np.array(data_dict_1['Reduced Chi^2'])[np.array(data_dict_1['Reduced Chi^2']) < 5.],np.array(data_dict_1['source']['g']['n_sersic'])[np.array(data_dict_1['Reduced Chi^2']) < 5.],label='Lens Candidates')
ax[27].scatter(np.array(data_dict_2['Reduced Chi^2'])[np.array(data_dict_2['Reduced Chi^2']) < 5.],np.array(data_dict_2['source']['g']['n_sersic'])[np.array(data_dict_2['Reduced Chi^2']) < 5.],label='Ring Galaxies')
ax[27].set_title('g band: \n n (source) vs Chi^2')
ax[27].legend()


ax[28].set_axis_off()
ax[29].set_axis_off()


# plt.subplot(5,5,2)
f.tight_layout()
f.savefig(results_path + '/histograms_Full',dpi = 200)
plt.close(f)
# plt.show()




    
    
    








