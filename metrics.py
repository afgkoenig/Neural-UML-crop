import os
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon
from sklearn import metrics
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter


dim = 128 #64
K_fold = 5

#calcul the f1_score : if the distance between predict center and real center is inferior to a delta, we consider the prediction to be valid
def f1_score(label,pred,delta,shape):
    centre = np.array([float(label[1])*shape[1],float(label[2])*shape[0]])
    centre2 = np.array([float(pred[1])*shape[1],float(pred[2])*shape[0]])
    if np.linalg.norm(centre-centre2)<delta : return 1
    else : return 0


#changes the data storage format
 
# the data is : class x_center y_center width height
# center (0;0) top left of the image
# after
# the data is : class;x1;y1;x2;y2
# center (0;0) top left of the image

def centre_to_corner(elt,shape):
    elt[1] = float(elt[1]) * shape[1]
    elt[2] = float(elt[2])* shape[0]
    elt[3] = float(elt[3]) * shape[1]
    elt[4] = float(elt[4]) * shape[0]
    x = elt[1]
    y = elt[2]
    width = elt[3]/2
    height = elt[4]/2
    elt[1] = x - width 
    elt[2] = y - height
    elt[3] = x + width
    elt[4] = y + height
    return elt

# calcul the recovery percentage
def calcul_intersect(elt1,elt2,shape):
    elt1 = centre_to_corner(elt1,shape)
    elt2 = centre_to_corner(elt2,shape)
    rect1 = Polygon(((elt1[1],elt1[2]),(elt1[3],elt1[2]),(elt1[3],elt1[4]),(elt1[1],elt1[4])))
    rect2 = Polygon(((elt2[1],elt2[2]),(elt2[3],elt2[2]),(elt2[3],elt2[4]),(elt2[1],elt2[4])))
    inter = rect1.intersection(rect2).area
    return (inter/rect1.area)*100

def cm_kfold_analysis(list_cm, classes, ax=None, filename="cm_kfolds", figsize=(17,17)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      classes:   aliases for the labels. String array to be shown in the cm plot.
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    
    sns.set(font_scale=1.5)
    
    cm = np.array(list_cm)*100
    cm_mean = np.mean(cm, axis=0)
    
    cm_std = np.std(cm, axis=0)
    
    annot = np.zeros(cm_mean.shape, dtype='<U20')
    nrows, ncols = cm_mean.shape
    for i in range(nrows):
        for j in range(ncols):
            annot[i, j] = f'{cm_mean[i,j]:.2f}%\n$\pm$ {cm_std[i,j]:.2f}'
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
        plt.yticks(va='center')

        sns.heatmap(cm_mean, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=True, cbar_kws={'format':PercentFormatter()}, yticklabels=classes, cmap="Blues")
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        if filename:
            plt.savefig(filename,  bbox_inches='tight')
    else:
        sns.heatmap(cm_mean, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=True, cbar_kws={'format':PercentFormatter()}, yticklabels=classes, cmap="Blues")
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.yticks(va='center')
    
    sns.reset_orig()


def main(nmb):

    liste_pixel = np.arange(0,11,0.1)
    liste_f1 = np.zeros((7,len(liste_pixel)))
    liste_intersect = []
    prediction = []
    reality = []
    count_non_prediction = np.zeros((7))
    
    # predict on image cropped

    model = YOLO(f"./proto_{nmb}/train/weights/best.pt") # choose your model
    Path_test = f"./images/test{nmb}/" # image to visualize
    pred = model.predict(Path_test,save=True,save_txt = True,conf=0.3,save_conf=True,device = 0,max_det = 1)

    try :
        os.rename("./runs/detect/predict/",f"./runs/detect/predict{nmb}/")
    except : FileExistsError

    for files in os.listdir(f"./labels/test{nmb}"):
        with open(f"./labels/test{nmb}/"+files, "r") as filereader:
            for row in filereader:
                row = row.split(" ")
                try:
                    with open(f"./runs/detect/predict{nmb}/labels/"+files, "r") as filereader2:
                        row2 = filereader2.readlines()
                        for elt in row2:
                            elt = elt.split(" ")
                            prediction.append(int(elt[0]))
                            reality.append(int(row[0]))
                            if int(row[0]) == int(elt[0]): #if we have the same label we calculate the f1_score and the recovery percentage
                                shape = np.shape(Image.open(f"./images/test{nmb}/"+files[:-4]+".jpg"))
                                for i in range (len(liste_pixel)):
                                    liste_f1[int(row[0]),i]+=f1_score(row,elt,liste_pixel[i],shape)
                                liste_intersect.append(calcul_intersect(row,elt,shape))

                except: 
                    count_non_prediction[int(row[0])] +=1


    confusion_matrix = metrics.confusion_matrix(reality, prediction)

    count_TRUE = np.sum(np.fromiter((confusion_matrix[i,i] for i in range (7)),int))
    count_global = np.sum(confusion_matrix) + np.sum(count_non_prediction)


    confusion_matrix_normalized = metrics.confusion_matrix(reality, prediction,normalize="true")

    cm_kfold_analysis([confusion_matrix_normalized],classes=["navigable","inheritance","realization","dependency","aggregation","composition","classe"],filename=f"confusion_matrix_{nmb}")

    plt.figure(nmb,clear=True,figsize=(5,5))

    array = []

    for i in range(7):
        precision = liste_f1[i]/ (liste_f1[i] + np.sum(confusion_matrix[:,i]) - confusion_matrix[i,i]) #FP
        recall = liste_f1[i] / (np.sum(confusion_matrix[i,:]) + count_non_prediction[i]) #TP_proche/(TP_proche+FN+TP_loin+non_prediction) mais np.sum contient TP donc FN-TP + (TP-TP_proche) donc juste TP_proche/(FN+non_prediction)
        copie = 2*precision*recall/(precision+recall)
        plt.plot(liste_pixel,copie,lw=3)
        array.append(precision)
        array.append(recall)

    array= np.array(array).T
    df = pd.DataFrame(array,index = liste_pixel , columns= ["navigable_precision", "navigable_recall","inheritance_precision", "inheritance_recall","realization_precision", "realization_recall","dependency_precision", "dependency_recall","aggregation_recision", "aggregation_recall", "composition_precision", "composition_recall", "classe_precision", "classe_recall"])
    df.to_csv(f"./results_{nmb}.csv")
        
        
    plt.legend(["navigable","inheritance","realization","dependency","aggregation","composition","classe"],loc="lower right",fontsize=20)
    plt.grid(lw=2)
    plt.xlabel("Distance (pixels)",fontsize=20)
    plt.ylabel("F measure (%)",fontsize=20)
    plt.ylim([0,1])
    plt.savefig(f"f1_score_{nmb}")


    return (count_TRUE/count_global)*100, np.mean(np.array(liste_intersect)) , reality, prediction, confusion_matrix_normalized




liste_acc = []
liste_intersection = []
liste_reality = []
liste_pred = []
liste_confusion_matrix = []
for i in range(1,K_fold+1):
    results = main(i)
    liste_acc.append(results[0])
    liste_intersection.append(results[1])
    liste_reality.append(results[2])
    liste_pred.append(results[3])
    liste_confusion_matrix.append(results[4])
print("acc : " , liste_acc)
print("IoU : " , liste_intersection)
mean = np.mean(np.array(liste_acc))
std = np.sqrt(np.sum((liste_acc-mean)**2)/K_fold)
print("mean : " ,mean)
print("std : " , std)



cm_kfold_analysis(liste_confusion_matrix,classes=["navigable","inheritance","realization","dependency","aggregation","composition","classe"])