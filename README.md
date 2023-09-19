# thesis
first run utils_NN.py search1Seg(directory=directory where data signals are)

run utils_NN.py train_val_set(directory=directory where data signals are for train and validation)

run utils_NN.py test_set(directory=directory where data signals are for test)


propuesta uses 3 representations of STFT

propuesta6 uses 3 representations of Melspectrograms


in test_CNN.py you need to change 

new_model = path of model to test
trns_indx = 0 for STFT, 5 for melspectrograms
