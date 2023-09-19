
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib.pyplot as plt
import utils_NN as ut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def metricas(y, z):

    #z = np.asarray(z).squeeze()
    
    cm = confusion_matrix_(y, z)
    
    TP = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TN = cm[1,1]
    
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    Fsc = ( 2 * Precision * Recall ) / ( Precision + Recall )
    
    all_events =  (TP+FP+FN+TN)
    
    Efficiency = (TP + TN)/ all_events
    FAR = FN / all_events
    
    return cm, Fsc, Precision, Recall, Efficiency, FAR

# Confusion matrix

def confusion_matrix_(y, z):
    
    m = y.shape[0]
    c = y.shape[1]
    
    y = np.argmax(y, axis=1)
    
    z = np.argmax(z, axis=1)
   

    cm = confusion_matrix(y, z)
    ConfusionMatrixDisplay.from_predictions(y, z, normalize='true')
    plt.show()
    return cm 


def main():
    
    # choose the transform to use trns_indx
    # 0 - STFT
    # 1 - Morlet transform
    # 2 - Discrete wavelet
    # 3 - Continuous wavelet T
    # 4 - fast Continuous WT
    # 5 - Melspectrogram
    # List of names corresponding to each transformation
    redes_names = ['STFT','WT_Morlet','DWT','','','MELSPECTROGRAM',
                   'PROPOSAL V1','PROPOSAL V2','PROPOSAL V3','PROPOSAL V4','PROPOSAL V5']
    dist = [
             '0.1',
             '2.71',
             '5.05',
             '7.39',
             '10'
             ]
    dist__ = [   0.1,
                2.71,
                5.05,
                7.39,
                10
    ]
    
    comp = True
    trns_indx = 0
    F = []
    E = []
    FA= []
    
    new_model = keras.models.load_model('redes/proposal_alldist.h5')
    
    for dist_ in dist:
        print('test dist: ',dist_)
        
        X_test_filenames = np.load('data/samples_names/test_list_dist_'+dist_+'.npy')
        y_test = np.load('data/samples_names/test_labels_dist_'+dist_+'.npy')
        
        batch_size = 32
        
        my_test_batch_generator = ut.My_Custom_Generator(X_test_filenames, y_test, 
                                                               batch_size, 
                                                               trns_indx=trns_indx,
                                                               comp = comp)
        
        # Evaluate the model on the test data using `evaluate`
        # print("Evaluate on test data")
        # results = new_model.evaluate(my_validation_batch_generator)
        # print("test loss, test acc, False positive:", results)
        
        #generate predictions
        print("Generate predictions")
        predictions = new_model.predict(my_test_batch_generator)
        
        cm, Fsc, Pr, Re, Ef, FAR = metricas(y_test, predictions)
        plt.show()  
        
        print('Fscores: ', Fsc)
        print('Precision: ', Pr)
        print('Recall: ', Re)
        print('Efficiency: ', Ef )
        print('FAR: ', FAR,'\n')
        
    
        F.append(Fsc)
        E.append(Ef)
        FA.append(FAR)
        
    plt.plot(dist__,F, color='magenta', marker='o',mfc='pink' )
    plt.ylabel('Fscores') #set the label for y axis
    plt.xlabel('Dist')
    plt.show()
    
    plt.plot(dist__,E, color='magenta', marker='o',mfc='pink' )
    plt.ylabel('Efficiency') #set the label for y axis
    plt.xlabel('Dist')
    plt.show()
    
    plt.plot(dist__,FA, color='magenta', marker='o',mfc='pink' )
    plt.ylabel('FAR') #set the label for y axis
    plt.xlabel('Dist')
    plt.show()
        
if __name__ == '__main__':   
	 main()


