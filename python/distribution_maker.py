# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:07:27 2016

@author: wvandepo
"""
import make_ps_distribution as dis

dis.get_gauss_distribution(output='../input/initial_distribution_gauss',
                           input='twiss_thin_nom.tfs',n_part=1000, sigmas=4,
                           beam_t='FT', seed=254)
