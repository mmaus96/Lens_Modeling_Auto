kwargs_lens_sigma = []
kwargs_lower_lens = []
kwargs_upper_lens = []

kwargs_lens_sigma.append({'theta_E': .01, 'e1': 0.01, 'e2': 0.01,
                         'center_x': 0.001, 'center_y': 0.001})
kwargs_lower_lens.append({'theta_E': 1.0, 'e1': -0.2, 'e2': -0.2, 'center_x': -0.5, 'center_y': -0.5})
kwargs_upper_lens.append({'theta_E': 2.0, 'e1': 0.2, 'e2': 0.2, 'center_x': 0.5, 'center_y': 0.5})

kwargs_lens_sigma.append({'gamma1': 0.1, 'gamma2': 0.1})
kwargs_lower_lens.append({'gamma1': -0.2, 'gamma2': -0.2})
kwargs_upper_lens.append({'gamma1': 0.2, 'gamma2': 0.2})

kwargs_source_sigma = []
kwargs_lower_source = []
kwargs_upper_source = []

kwargs_source_sigma.append({'R_sersic': 0.01,'n_sersic': 0.01, 'center_x': 0.001, 'center_y': 0.001})
kwargs_lower_source.append({ 'R_sersic': 0.001, 'n_sersic': .01, 'center_x': -0.5, 'center_y': -0.5})
kwargs_upper_source.append({ 'R_sersic': 2.0, 'n_sersic': 4., 'center_x': 0.5, 'center_y': 0.5})

kwargs_lens_light_sigma = []
kwargs_lower_lens_light = []
kwargs_upper_lens_light = []

kwargs_lens_light_sigma.append({'R_sersic': 0.01, 'n_sersic': 0.01,  'center_x': 0.001, 'center_y': 0.001})
kwargs_lower_lens_light.append({'R_sersic': 0.001, 'n_sersic': 0.1, 'center_x': -0.5, 'center_y': -0.5})
kwargs_upper_lens_light.append({'R_sersic': 5., 'n_sersic': 5., 'center_x': 0.5, 'center_y': 0.5})
