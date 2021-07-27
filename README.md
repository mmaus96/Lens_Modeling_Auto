# Lens_Modeling_Auto
Automated strong lens modeling script using Lenstronomy package

The main front end script for modeling: 'main_modeling_script.py'
Backend code used by the script is in 'auto_modeling_functions.py', 'fit_sequence_functions.py', and 'plot_functions.py'. 
The only thing that needs to be changed by the user in the backend scripts is in the 'auto_modeling_functions.py', line 35, where the 'path_to_script' 
variable needs to be the full file path to the downloaded Lens_Modeling_Auto folder containing all of the scripts (e.g. path_to_script= '/Users/mmaus/Desktop/.../Lens_Modeling_Auto/')

In the 'main_modeling_script.py' script, the user must fill out all variables in the User Inputs section at the top of the script. Refer to the 'DES_modeling_script.py' and 
'CFIS_modeling_script.py' for examples of how these variables were defined for actual data.

In order to create the mosaics plots such as those shown in the papers, use the 'results_mosaic.py' script. 
