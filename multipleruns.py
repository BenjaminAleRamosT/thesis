# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 14:17:39 2023

@author: benja
"""

import CNN_Astone as ast
import CNN_AstonefCwt as astfcwt
import numpy as np

def main(trns_indx = 0):
    
    # choose the transform to use trns_indx
    # 0 - STFT
    # 1 - Morlet transform
    # 2 - Discrete wavelet
    # 3 - Continuous wavelet T
    # 4 - fast Continuous WT
    # 5 - Melspectrogram
    
    dist = ['0.1','2.71', '5.05','7.39', '10']
    
    for dist_ in dist:
        for trns_indx in [0,1,2,5]:
            print(trns_indx)
            if trns_indx == 4:
                    
                astfcwt.main()
            
            else:
                n = [4,6,3,5]
                
                filters = np.ones(n[trns_indx])*8
                
                ast.main(
                    directory = 'data/samples_names',
                    dist = dist_,
                    trns_indx = trns_indx,
                    fmax = 2048,
                    filters = filters,
                    dropout = 1,  # 0 = soft-dropout, 1 = dropout
                    #dropout params
                    p = 0.3,
                    a = 2, b = 5
                )
        
if __name__ == "__main__":
   main()    
    
    