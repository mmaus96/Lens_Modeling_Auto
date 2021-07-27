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
from math import ceil
import numpy as np
import pandas as pd
from Lens_Modeling_Auto.auto_modeling_functions import openFITS
from Lens_Modeling_Auto.auto_modeling_functions import find_components
from Lens_Modeling_Auto.auto_modeling_functions import calcBackgroundRMS
from Lens_Modeling_Auto.auto_modeling_functions import prepareData
from Lens_Modeling_Auto.auto_modeling_functions import prepareFit
from Lens_Modeling_Auto.auto_modeling_functions import find_components
from Lens_Modeling_Auto.auto_modeling_functions import mask_for_sat
from Lens_Modeling_Auto.auto_modeling_functions import find_lens_gal
from functools import reduce
from matplotlib.colors import SymLogNorm
import re
from matplotlib.patches import Circle
from copy import deepcopy
from lenstronomy.Plots.model_plot import ModelPlot


# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/ringcatalog/'
# im_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/ringcatalog/data/'
# csv_paths = [path + 'results_full_catalog/']
# results_path = path + 'results_full_catalog'

# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Sure_Lens/'
# im_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Sure_Lens/data/'
# csv_paths = [path + 'results_new_priors/']
# results_path = path + 'results_new_priors'

# path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_lenses/Sure_Lens/'
# im_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_lenses/data/'
# csv_path = path + 'results_PEMD_Ap11/'
# results_path = path + 'results_PEMD_Ap11'

path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_deblended/deblended_image_2/modeling_results_deblended/'
im_path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/CFIS_deblended/deblended_image_2/originals/lenses/'
csv_path = path + 'results_deblended/'
results_path = path + 'results_deblended'



if not exists(results_path):
    os.mkdir(results_path)

#Folder names for data, psf, noise map, original image [TO DO BY USER]
# band_list = ['g','r','i']
# obj_name_location = 0
# deltaPix = 0.27
# psf_upsample_factor = 1

band_list = ['r']
obj_name_location = 0
deltaPix = 0.1857
psf_upsample_factor = 2

ncols = 7

# zeroPt = 30
# numCores = 1


#Make dataframes from csv files
# df_list = []
# for i,x in enumerate(csv_paths):
#     df = pd.read_csv(x + 'full_results.csv',delimiter =',')
#     df_list.append(df.loc[:,:])    
    
df_final = pd.read_csv(csv_path + 'full_results.csv',delimiter =',')

nrows = ceil(len(df_final) / ncols)


for i in range(len(band_list)):
    fig,axes = plt.subplots(nrows,ncols,figsize = (ncols*5,nrows*5))
#     fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    ax = axes.ravel()
#     for j in range(len(df_final)):
    for j in range(len(df_final)):
        fn = df_final.loc[j,'FITS filename']
        print(im_path + fn)
        ID = df_final.loc[j,'ID']
        data,_ = openFITS(im_path + fn)
        mask_px = float(df_final.loc[j,'Mask Size (Pixels)'])
        mask_as = float(df_final.loc[j,'Mask Size (Arcsec)'])
        

#         image = data[0][i]
#         image = data[i]


#         plt.subplot(ceil(len(df_final) / ncols),ncols,j+1)
        ax[j].imshow(data[i], origin='lower', norm = SymLogNorm(5))
        ax[j].set_title('Image {}: \n ID: {}'.format(j+1,ID),fontsize=15)
        
        
        numPix = len(data[i])
        
        c_x,c_y = find_lens_gal(data[-1],deltaPix,show_plot=False,title=ID)
        
#         (x_sats, y_sats), (new_center_x, new_center_y) = find_components(image, deltaPix,lens_rad_arcsec = lens_rad_arcsec, 
#                                                                          gal_rad_ratio = gal_rad_ratio, 
#                                                                          min_size_arcsec= min_size_arcsec,
#                                                                          lens_rad_ratio = lens_rad_ratio,
#                                                                          thresh=thresh, show_locations=False)
#         for k in range(len(x_sats)):
#             plt.scatter([x_sats[k]], [y_sats[k]], c='red', s=100, marker='.')
#         ax[j].scatter(c_x, c_y,c='red', s=100, marker='*')
        draw_lens_circle = Circle((c_x, c_y),int(mask_px) ,fill=False)
#         draw_gal_circle = Circle((new_center_x, new_center_y),gal_rad, fill = False)
#         ax[j].add_artist(draw_lens_circle)
#         plt.gcf().gca().add_artist(draw_gal_circle)
        ax[j].get_xaxis().set_visible(False)
        ax[j].get_yaxis().set_visible(False)        
#         ax[j].text(1,1,'r = {:.2f}"'.format(mask_as),fontweight='bold',color='red')
    for l,a in enumerate(ax):
        if l >= len(df_final):
            a.set_axis_off()
    fig.tight_layout()
#     fig.savefig(results_path + '/{}_band_masks.png'.format(band_list[i]),dpi = 200)
    fig.savefig(results_path + '/{}_band_lenses.png'.format(band_list[i]),dpi = 200)
    plt.close(fig)






