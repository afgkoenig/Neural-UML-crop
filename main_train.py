import os
from PIL import Image
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    dim = 128 #64
    K_fold = 5
    global_count = 1
    seed=1
    EPOCH_number = 30
    name_csv = "label.csv"

    os.mkdir(f"./images")
    os.mkdir(f"./labels")

    for i in range (1,K_fold+1):

            
            os.mkdir(f"./images/dataset{i}")
            os.mkdir(f"./images/validation{i}")
            os.mkdir(f"./images/test{i}")

            os.mkdir(f"./labels/dataset{i}")
            os.mkdir(f"./labels/validation{i}")
            os.mkdir(f"./labels/test{i}")
         


    with open(name_csv, newline='') as csvfile:

        filereader = csv.reader(csvfile, delimiter=';')
        liste_stockage = [[],[],[],[],[],[],[]]

        for row in filereader:

            if row[1] == "Yes":

                name = row.pop(0)

                # we convert all image in RGB for be uniform

                img = Image.open(f"pre_traitement/images_vrac/{name}")
                if img.mode == "RGBA" and np.array(img)[0,0,3]==0 and np.array(img)[0,0,0]==0:
                    img = np.array(img)
                    img_final = img[:,:,:3]
                    img_final[:,:,0] += img[:,:,3]
                    img_final[:,:,1] += img[:,:,3]
                    img_final[:,:,2] += img[:,:,3]
                    img= Image.fromarray(img_final)
                else :
                    img = img.convert("RGB")


                row.pop(0)
                length = len(row)-1
                row.pop(length)


                row = [float(elt) for elt in row]
                row = np.reshape(row,(int(length/5),5))
                
                # division par 2 à cause du resize du fichier éxécutable
                row[:,:-1] = row[:,:-1]/2   
                

                # the data is : x1;y1;x2;y2;class
                # center (0;0) top left of the image

                # convert csv files into txt files using YOLO's advanced data organisation system
                
                # the data is : class x_center y_center width height
                # center (0;0) top left of the image
                
                width = abs(row[:,0] - row[:,2])
                height = abs(row[:,1] - row[:,3])

                row[:,0] = width/2 + np.minimum(row[:,0],row[:,2])
                row[:,1] = height/2 + np.minimum(row[:,1],row[:,3])
                row[:,2] = width
                row[:,3] = height

                # we have some big images so we create cropped image to pass at yolo
                for elt in row:

                    crop_width = elt[2] + dim - elt[2]%dim
                    crop_height = elt[3] + dim - elt[3]%dim
                    top_left_x = elt[0] - crop_width/2
                    top_left_y = elt[1] - crop_height/2
                    bottom_right_x = elt[0] + crop_width/2
                    bottom_right_y = elt[1] + crop_height/2
                    im_crop = img.crop((top_left_x,top_left_y,bottom_right_x,bottom_right_y))
                    texte = str(int(elt[4]))+" "+str((bottom_right_x-top_left_x)/(2*crop_width))+" "+str((bottom_right_y-top_left_y)/(2*crop_height))+" "+str(elt[2]/crop_width)+" "+str(elt[3]/crop_height)
                    liste_stockage[int(elt[4])].append((im_crop,texte,global_count))
                    global_count+=1
                    
    count_global = []

    # create k dataset for reproducible results

    for classe in liste_stockage:
        len_folder = int(len(classe)/K_fold)
        random.seed(seed)
        random.shuffle(classe)
        count_global.append(len(classe))
        liste_numero = ["1 0 0"]*(K_fold-2) + ["0 1 0"] + ["0 0 1"]
        for i in range (1,K_fold+1):
            for j in range (K_fold):
                for k in range (len_folder):
                    if int(liste_numero[j][0]) == 1:
                        classe[k+j*len_folder][0].save(f"./images/dataset{i}/{classe[k+j*len_folder][2]}.jpg")
                        with open (f"./labels/dataset{i}/{classe[k+j*len_folder][2]}.txt","w") as file:
                            file.write(classe[k+j*len_folder][1])


                    if int(liste_numero[j][2]) == 1:
                        classe[k+j*len_folder][0].save(f"./images/validation{i}/{classe[k+j*len_folder][2]}.jpg")
                        with open (f"./labels/validation{i}/{classe[k+j*len_folder][2]}.txt","w") as file:
                            file.write(classe[k+j*len_folder][1])
                    if int(liste_numero[j][4]) == 1:
                        classe[k+j*len_folder][0].save(f"./images/test{i}/{classe[k+j*len_folder][2]}.jpg")
                        with open (f"./labels/test{i}/{classe[k+j*len_folder][2]}.txt","w") as file:
                            file.write(classe[k+j*len_folder][1])

            num = liste_numero.pop(0)
            liste_numero.append(num)
            texte = f"path: ../../  # dataset root dir \ntrain: test_V4/images/dataset{i}  # train images (relative to 'path') \nval: test_V4/images/validation{i}  # val images (relative to 'path') \ntest: test_V4/images/test{i} # test images (optional) \n# Classes \nnames: \n  0: navigable \n  1: inheritance \n  2: realization \n  3: dependency \n  4: aggregation \n  5: composition \n  6: classe"
            with open(f"./train{i}.yaml","w") as yaml:
                yaml.write(texte)



    liste_data = ["navigable","inheritance","realization","dependency","aggregation","composition","classe"]
    plt.bar(range(len(count_global)),count_global,tick_label=liste_data)
    plt.xticks(rotation=30, ha='right')
    plt.savefig("./hist.png",bbox_inches='tight')

    # training
    
    for number in range (1,K_fold+1):
        model = YOLO('yolov8n.pt')

        model.train(data=f"./train{number}.yaml", epochs=EPOCH_number , save_period = 1,val = True,seed = seed,device=0, project = f"proto_{number}")
  
    

if __name__ == '__main__':
    main()