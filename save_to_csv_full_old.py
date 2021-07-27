import pandas as pd
import numpy as np
from lenstronomy.LightModel.light_param import LightParam

num = it + 1 #it is main image iteration, num is [it] shifted so it starts with 1 instead of 0
shapelet_kwargs = LightParam(['SHAPELETS'],kwargs_fixed = [{}])._param_name_list[0]
SERSIC_ELL_kwargs = LightParam(['SERSIC_ELLIPSE'],kwargs_fixed = [{}])._param_name_list[0]

#################### Write to full results csv ############################
data_list_full = [[data_pairs_dicts[it]['image_data'],data_pairs_dicts[it]['object_ID'],
                   data_pairs_dicts[it]['RA'],data_pairs_dicts[it]['DEC'],red_X_squared]]
for x in kwargs_result['kwargs_lens']:
    data_list_full.append(list(x.values()))
    
for x in kwargs_result['kwargs_source']:
    if not 'SERSIC_ELLIPSE' in source_model_list:
        data_list_full.append(['N/A'] * len(SERSIC_ELL_kwargs))
    data_list_full.append(list(x.values()))
    if not 'SHAPELETS' in source_model_list:
        data_list_full.append(['N/A'] * len(shapelet_kwargs))
    
    
for x in kwargs_result['kwargs_lens_light']:
    data_list_full.append(list(x.values()))
data_flat_full = [item for sublist in data_list_full for item in sublist]
data_array_full = np.array([data_flat_full])


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
    if not 'SERSIC_ELLIPSE' in multi_source_model_list:
        cols_full[0].extend(np.array(['{} Band: SERSIC_source'.format(bands_for_source_headers[i])]*len(SERSIC_ELL_kwargs)))
        cols_full[1].extend(np.array(SERSIC_ELL_kwargs))
    
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

#df = pd.DataFrame(data_array_full,index= [['Image_{}'.format(num)],[data_pairs[it][0]]], columns=levels_full)
df = pd.DataFrame(data_array_full,index= [['Image_{}'.format(num)]], columns=levels_full)

print('\n')
print('full data frame: ', df)


# if it == 0:
#     df.to_csv(csv_path + '/full_results.csv')   #creates the csv on after modeling first image
# else:
df.to_csv(csv_path + '/full_results.csv',mode='a',header = False)  #adds the df to existing csv
    