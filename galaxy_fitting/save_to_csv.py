import pandas as pd
import numpy as np
from lenstronomy.LightModel.light_param import LightParam

num = it + 1 #it is main image iteration, num is [it] shifted so it starts with 1 instead of 0
SERSIC_ELL_kwargs = LightParam(['SERSIC_ELLIPSE'],kwargs_fixed = [{}])._param_name_list[0]
data_list_full = [[data_pairs_dicts[it]['image_data'],red_X_squared]]

for x in kwargs_result['kwargs_lens_light']:
    data_list_full.append(list(x.values()))
data_flat_full = [item for sublist in data_list_full for item in sublist]
data_array_full = np.array([data_flat_full])

cols = [['',''],['FITS filename','reduced chi^2']]

multi_source_model_list = []
multi_lens_light_model_list = []

for i in range(len(kwargs_data)):
    multi_source_model_list.extend(deepcopy(source_model_list))
    multi_lens_light_model_list.extend(deepcopy(lens_light_model_list))

model_kwarg_names = get_kwarg_names(lens_model_list,multi_source_model_list,
                                     multi_lens_light_model_list,kwargs_fixed)

bands_for_lens_light_headers = []

for b in band_list:
    bands_for_lens_light_headers.extend([b for i in lens_light_model_list])
    
for i in range(len(multi_lens_light_model_list)):
    cols[0].extend(np.array(['{} Band: {}_lens_light'.format(bands_for_lens_light_headers[i],multi_lens_light_model_list[i])]
                                 *len(model_kwarg_names['kwargs_lens_light'][i])))
    cols[1].extend(np.array(model_kwarg_names['kwargs_lens_light'][i]))

tuples = list(zip(*cols))
levels = pd.MultiIndex.from_tuples(tuples)

df_data = pd.DataFrame(data_array_full,index= [['Image_{}'.format(num)]], columns=levels)

print('\n')
print('full data frame: ', df_data)

df_data.to_csv(csv_path + '/full_results.csv',mode='a',header = False)  #adds the df to existing csv
