import pandas as pd
import numpy as np
from lenstronomy.LightModel.light_param import LightParam



num = it + 1 #it is main image iteration, num is [it] shifted so it starts with 1 instead of 0

#################### Write to lens results csv ############################
data_list = [[data_pairs_dicts[it]['image_data'],data_pairs_dicts[it]['object_ID'],
              data_pairs_dicts[it]['RA'],data_pairs_dicts[it]['DEC'],image_model_time,red_X_squared,mask_dict_list[0]['size arcsec'],mask_dict_list[0]['size pixels']]]
for x in kwargs_result['kwargs_lens']:
    data_list.append(list(x.values()))
data_flat = [item for sublist in data_list for item in sublist]
data_array = np.array([data_flat])

print('data array:', data_array)
cols = [['FITS filename','ID','RA','DEC','Modeling Time (min)','reduced chi^2','Mask Size (Arcsec)','Mask Size (Pixels)']]
for i in range(len(lens_model_list)):
    for name in np.array(model_kwarg_names['kwargs_lens'][i]):
        cols[0].append('{}_lens.{}'.format(lens_model_list[i],name))
#         cols[1].extend(np.array(model_kwarg_names['kwargs_lens'][i]))

tuples = list(zip(*cols))
levels = pd.MultiIndex.from_tuples(tuples)


lens_df = pd.DataFrame(data_array,index= [['Image_{}'.format(num)]], columns=levels)

print('\n')
print('lens data frame: ', lens_df)
# if it == 0:
#     lens_df.to_csv(csv_path + '/lens_results.csv')  #creates the csv on after modeling first image
# else:

lens_df.to_csv(csv_path + '/lens_results.csv',mode='a',header = False) #add df to existing csv
    


    