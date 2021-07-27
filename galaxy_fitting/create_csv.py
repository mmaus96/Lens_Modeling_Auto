import pandas as pd
import numpy as np
from lenstronomy.LightModel.light_param import LightParam


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

df = pd.DataFrame([],index= [], columns=levels)
df.to_csv(csv_path + '/full_results.csv')