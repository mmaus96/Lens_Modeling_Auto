import pandas as pd
import numpy as np
from lenstronomy.LightModel.light_param import LightParam


shapelet_kwargs = LightParam(['SHAPELETS'],kwargs_fixed = [{}])._param_name_list[0]
SERSIC_ELL_kwargs = LightParam(['SERSIC_ELLIPSE'],kwargs_fixed = [{}])._param_name_list[0]
ps_kwargs = ['ra_source','dec_source','point_amp']

################## CSV File for Lens Model Results ##############################

cols = [['FITS filename','ID','RA','DEC','Modeling Time (min)','reduced chi^2','Mask Size (Arcsec)','Mask Size (Pixels)']]
for i in range(len(lens_model_list)):
    for name in np.array(model_kwarg_names['kwargs_lens'][i]):
        cols[0].append('{}_lens.{}'.format(lens_model_list[i],name))

tuples = list(zip(*cols))
levels = pd.MultiIndex.from_tuples(tuples)


lens_df = pd.DataFrame([],index= [], columns=levels)

lens_df.to_csv(csv_path + '/lens_results.csv')  #creates the csv 



################### CSV FIle for Full Model Results ##############################

cols_full = [['FITS filename','ID','RA','DEC','Modeling Time (min)','reduced chi^2','Mask Size (Arcsec)','Mask Size (Pixels)']]
for i in range(len(lens_model_list)):
    for name in np.array(model_kwarg_names['kwargs_lens'][i]):
        cols_full[0].append('{}_lens.{}'.format(lens_model_list[i],name))

bands_for_source_headers = []
bands_for_lens_light_headers = []
bands_for_ps_headers = []

for b in band_list:
    bands_for_source_headers.extend([b for i in source_model_list])
    bands_for_lens_light_headers.extend([b for i in lens_light_model_list])
    if point_source_model_list !=None:
        bands_for_ps_headers.extend([b for i in point_source_model_list])

for i in range(len(multi_source_model_list)):
    for name in model_kwarg_names['kwargs_source'][i]:
        cols_full[0].append('{} Band: {}_source.{}'.format(bands_for_source_headers[i],multi_source_model_list[i],name))
    if not 'SERSIC_ELLIPSE' in multi_source_model_list:
        for name in SERSIC_ELL_kwargs:
            cols_full[0].append('{} Band: SERSIC_source.{}'.format(bands_for_source_headers[i],name))
    
#     cols_full[1].extend(np.array(model_kwarg_names['kwargs_source'][i]))
    
    if not 'SHAPELETS' in multi_source_model_list:
        for name in shapelet_kwargs:
            cols_full[0].append('{} Band: SHAPELETS_source.{}'.format(bands_for_source_headers[i],name))
    
                                 
for i in range(len(multi_lens_light_model_list)):
    for name in model_kwarg_names['kwargs_lens_light'][i]:
        cols_full[0].append('{} Band: {}_lens_light.{}'.format(bands_for_lens_light_headers[i],multi_lens_light_model_list[i],name))
        
if point_source_model_list !=None:
    for i in range(len(multi_ps_model_list)):
        for name in ps_kwargs:
            cols_full[0].append('{} Band: {}_point_source.{}'.format(bands_for_ps_headers[i],
                                                                     multi_ps_model_list[i],name))
        

tuples_full = list(zip(*cols_full))
levels_full = pd.MultiIndex.from_tuples(tuples_full)

#df = pd.DataFrame(data_array_full,index= [['Image_{}'.format(num)]], columns=levels_full)
df = pd.DataFrame([],index= [], columns=levels_full)

df.to_csv(csv_path + '/full_results.csv')   #creates the csv 


