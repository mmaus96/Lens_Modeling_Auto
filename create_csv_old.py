import pandas as pd
import numpy as np
from lenstronomy.LightModel.light_param import LightParam


shapelet_kwargs = LightParam(['SHAPELETS'],kwargs_fixed = [{}])._param_name_list[0]

################## CSV File for Lens Model Results ##############################

cols = [['','','','',''],['FITS filename','ID','RA','DEC','reduced chi^2']]
for i in range(len(lens_model_list)):
    cols[0].extend(np.array(['{}_lens'.format(lens_model_list[i])]*len(model_kwarg_names['kwargs_lens'][i])))
    cols[1].extend(np.array(model_kwarg_names['kwargs_lens'][i]))

tuples = list(zip(*cols))
levels = pd.MultiIndex.from_tuples(tuples)


lens_df = pd.DataFrame([],index= [], columns=levels)

lens_df.to_csv(csv_path + '/lens_results.csv')  #creates the csv 



################### CSV FIle for Full Model Results ##############################

cols_full = [['','','','',''],['FITS filename','ID','RA','DEC','reduced chi^2']]

for i in range(len(lens_model_list)):
    cols_full[0].extend(np.array(['{}_lens'.format(lens_model_list[i])]*len(model_kwarg_names['kwargs_lens'][i])))
    cols_full[1].extend(np.array(model_kwarg_names['kwargs_lens'][i]))

bands_for_source_headers = []
bands_for_lens_light_headers = []

for b in band_list:
    bands_for_source_headers.extend([b for i in source_model_list])
    bands_for_lens_light_headers.extend([b for i in lens_light_model_list])

for i in range(len(multi_source_model_list)):
    cols_full[0].extend(np.array(['{} Band: {}_source'.format(bands_for_source_headers[i],multi_source_model_list[i])]*len(model_kwarg_names['kwargs_source'][i])))
    cols_full[1].extend(np.array(model_kwarg_names['kwargs_source'][i]))
    if not 'SHAPELETS' in multi_source_model_list:
        cols_full[0].extend(np.array(['{} Band: SHAPELETS_source'.format(bands_for_source_headers[i])]*len(shapelet_kwargs)))
        cols_full[1].extend(np.array(shapelet_kwargs))
                                 
for i in range(len(multi_lens_light_model_list)):
    cols_full[0].extend(np.array(['{} Band: {}_lens_light'.format(bands_for_lens_light_headers[i],multi_lens_light_model_list[i])]
                                 *len(model_kwarg_names['kwargs_lens_light'][i])))
    cols_full[1].extend(np.array(model_kwarg_names['kwargs_lens_light'][i]))

tuples_full = list(zip(*cols_full))
levels_full = pd.MultiIndex.from_tuples(tuples_full)

#df = pd.DataFrame(data_array_full,index= [['Image_{}'.format(num)]], columns=levels_full)
df = pd.DataFrame([],index= [], columns=levels_full)

df.to_csv(csv_path + '/full_results.csv')   #creates the csv 


