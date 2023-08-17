# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 14:17:39 2023

@author: benja
"""

import propuesta as prp

def main():
    
    dist = ['0.1','2.71', '5.05','7.39', '10']
    
    for dist_ in dist:
        prp.main(
            directory = 'data/samples_names',
            dist = dist_
        )
        
if __name__ == "__main__":
   main()    
    
    