
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib.pyplot as plt
import utils_NN as ut

def metricas(y, z):

    #z = np.asarray(z).squeeze()
    
    cm = confusion_matrix(y, z)
    
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

def confusion_matrix(y, z):
    
    m = y.shape[0]
    c = y.shape[1]
    
    y = np.argmax(y, axis=1)
    
    z = np.argmax(z, axis=1)
   
    cm = np.zeros((c,c))
    
    for i in range(m):
         cm[z[i] ,y[i]] += 1
    
    #cm = np.flip(cm)
      
    
    # cm_m = np.zeros((cm.shape[0], 2, 2)) #confusion matrix per class

    # for i in range(cm.shape[0]):
    #     cm_m[i,0,0] = cm[i,i] #TP
    #     cm_m[i,0,1] = np.sum(np.delete(cm[i,:], i, axis=0)) #FP
    #     cm_m[i,1,0] = np.sum(np.delete(cm[:,i], i, axis=0)) #FN
    #     cm_m[i,1,1] = np.sum(np.delete(np.delete(cm, i, axis=1),i , axis=0 )) #TN
    
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
    dist_ = [   0.1,
                2.71,
                5.05,
                7.39,
                10
    ]
    
    i = 9 #change this to change model to test
    
    comp=False
    
    if i > 5:
        comp=True
        trns_indx = 0
        if i >= 9:
            trns_indx = 0
    else:
        trns_indx = i
    F = []
    E = []
    FA= []
    for j in dist:
        print('test dist: ',j)
        
        new_model = keras.models.load_model('redes/last/proposal_dist-'+j+'.h5')
        
        X_val_filenames = np.load('data/samples_names/val_list_dist_'+'10'+'.npy')
        y_val = np.load('data/samples_names/val_labels_dist_'+'10'+'.npy')
        
        batch_size = 32
        
        my_validation_batch_generator = ut.My_Custom_Generator(X_val_filenames, y_val, 
                                                               batch_size, 
                                                               trns_indx=trns_indx,
                                                               comp = comp)
        
        # Evaluate the model on the test data using `evaluate`
        # print("Evaluate on test data")
        # results = new_model.evaluate(my_validation_batch_generator)
        # print("test loss, test acc, False positive:", results)
        
        #generate predictions
        print("Generate predictions")
        predictions = new_model.predict(my_validation_batch_generator)
       
        cm, Fsc, Pr, Re, Ef, FAR = metricas(y_val, predictions)
        # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False,True])
        # cm_display.plot(cmap='Blues', values_format='')
        # cm_display.ax_.set_title(redes_names[j])
        plt.show()  
        
        print('Fscores: ', Fsc)
        print('Precision: ', Pr)
        print('Recall: ', Re)
        print('Efficiency: ', Ef )
        print('FAR: ', FAR,'\n')
        
    
        F.append(Fsc)
        E.append(Ef)
        FA.append(FAR)
        
    plt.plot(dist_,F, color='magenta', marker='o',mfc='pink' )
    plt.ylabel('Fscores') #set the label for y axis
    plt.xlabel('Dist')
    plt.show()
    
    plt.plot(dist_,E, color='magenta', marker='o',mfc='pink' )
    plt.ylabel('Efficiency') #set the label for y axis
    plt.xlabel('Dist')
    plt.show()
    
    plt.plot(dist_,FA, color='magenta', marker='o',mfc='pink' )
    plt.ylabel('FAR') #set the label for y axis
    plt.xlabel('Dist')
    plt.show()
        
if __name__ == '__main__':   
	 main()


