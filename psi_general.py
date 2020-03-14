# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:29:14 2017

@author: sallylee1989
"""

import numpy as np
import pandas as pd
import math



def get_PSI(vec_1, vec_2, y_name, n_bins = 3):
    ''' vec_1, vec_2: should be the vector in pd.DataFrame
        y_name: name of target
        n_bins: # group
    '''
    # partitioned by same lengh
    tmp_range = vec_1[y_name].max() - vec_1[y_name].min()
    tmp_step = tmp_range / n_bins
    tmp_cut = [vec_1[y_name].min() + i * tmp_step for i in range(n_bins)]

    out_vec = vec_1
    out_compre = vec_2
    for i in range(len(tmp_cut)-1):    
        tmp_vec = (vec_1[y_name] >= tmp_cut[i]) * 1 * (vec_1[y_name] < tmp_cut[i+1])  * ( i + 1 )  
        tmp_compre = (vec_2[y_name] >= tmp_cut[i]) * 1 * (vec_2[y_name] < tmp_cut[i+1])  * ( i + 1 )  
        
        if  i == 0 :
            out_vec['rank'] = tmp_vec
            out_compre['rank'] = tmp_compre   
        else:
            out_vec['rank'] = out_vec['rank'] + tmp_vec
            out_compre['rank'] = out_compre['rank'] + tmp_compre

            if i == len(tmp_cut)-2: 
                tmp_vec = (vec_1[y_name] >= tmp_cut[i+1]) * ( i + 2 )
                tmp_compre = (vec_2[y_name] >= tmp_cut[i+1]) * ( i + 2 ) 
    
                out_vec['rank'] = out_vec['rank'] + tmp_vec
                out_compre['rank'] = out_compre['rank'] + tmp_compre  

    tmp_1 = out_vec.groupby('rank').count()[y_name].reset_index().rename(columns = { y_name : 'v1'})
    tmp_2 = out_compre.groupby('rank').count()[y_name].reset_index().rename(columns = {y_name : 'v2'})
    
    tmp_r = list(set(pd.concat([tmp_1['rank'], tmp_2['rank']]) ))
    tmp_r = pd.DataFrame(tmp_r, columns = {'rank':0})
    tmp_3 = pd.merge(tmp_r, tmp_1, how = 'left', on = 'rank').fillna(0)
    tmp_3 = pd.merge(tmp_3, tmp_2, how = 'left', on = 'rank').fillna(0)
    
    psi_vec = (tmp_3.v1 - tmp_3.v2) * np.log(tmp_3.v1 + 0.0001 / tmp_3.v2 + 0.0001)
    out_psi = psi_vec.sum()
    print('\n Final PSI: ', out_psi, '\n')
    return out_psi
    
    
    
'''   
# ---- TEST ---- 
t1 = pd.DataFrame( [43, 65, 3, 23, 8, 11, 887, 32, 4, 54, 2, 99], columns = {'y':0} );
t2 = pd.DataFrame( [9, 3, 1, 65, 23, 655, 87, 4, 22, 23, 33, 44], columns = {'y':0} );    
    
get_PSI(t1, t2, 'y', n_bins = 3)  

    
 '''   
  

