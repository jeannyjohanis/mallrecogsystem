import numpy as np
import math
import matplotlib.pyplot as plt
import PIL
import os as os
import os, os.path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import messagebox

def resize():
    #myImage and myImage_new dir change accordingly
    myImage = "ModifiedPhoto"
    myImage_new = "image_new"

    #count files with jpg extension
    image_count = len([f for f in os.listdir(myImage) if f.endswith(".jpg")])
    #resizing to 100*100 and grayscale
    basewidth = 100
    hsize = 100
    count = 0
    for i in range(3):
        for j in range(int(image_count/3)):
            image = PIL.Image.open(myImage + '\\' + str(i+1) + '_' + str(j+1) +'.jpg')
            image = image.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
            image = image.convert("L")
            image.save(myImage_new + '\\' + str(i+1) + '_' + str(j+1) +'.jpg')

def toMatrix():
    from PIL import Image
    myImage_new = "image_new"
    #count files with jpg extension
    image_count = len([f for f in os.listdir(myImage_new) if f.endswith(".jpg")])
    #initialize empty numpy matrix
    matrix = np.zeros(shape=(image_count,100*100))
    #store images into matrix
    for i in range(3):
        index=36*i
        for j in range(int(image_count/3)):
                image = PIL.Image.open(myImage_new + '\\' + str(i+1) + '_' + str(j+1) +'.jpg')
                data = image.getdata()
                data = np.matrix(data)
                matrix[index+j] = data
    #save matrix into text file
    #change dir accordingly
    np.savetxt('image_new\\image.txt', matrix)

def traintestsplit():
    X = np.loadtxt('image_new\\image.txt')
    y=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30, random_state = 0)
    return (X_train, X_test, y_train, y_test)

def pca(X_train,X_test):
    from sklearn.decomposition import PCA
    n_components = 70
    pca = PCA(n_components = n_components, svd_solver = 'randomized', whiten = True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return (X_train_pca,X_test_pca)

def pcaIm(X_train, predIm):
    from sklearn.decomposition import PCA
    n_components = 70
    pca = PCA(n_components = n_components, svd_solver = 'randomized', whiten = True).fit(X_train)
    predIm_pca = pca.transform(predIm)
    return (predIm_pca)

def svm(X_train_pca,y_train,X_test_pca):
    from sklearn import svm
    from sklearn.svm import SVC
    clf = svm.SVC(gamma=0.0001 , C=100)
    clf.fit(X_train_pca,y_train)
    predictions=clf.predict(X_test_pca)
    return (predictions)

def confusionMatrix(y_test, predictions):
    conMat=confusion_matrix(y_test, predictions)
    return (conMat)

def PrecisionRecall(y_test,predictions):
    Mall=["AeonMall","MelakaMall", "MydinMall"]
    report=classification_report(y_test, predictions,target_names=Mall)
    score=accuracy_score(y_test,predictions)
    return(report,score)

def clicked1():

    #train model
    from PIL import Image
    resize()
    toMatrix()
    X_train, X_test, y_train, y_test = traintestsplit()
    X_train_pca,X_test_pca=pca(X_train,X_test)
    predictions=svm(X_train_pca,y_train,X_test_pca)
    conMat=confusionMatrix(y_test,predictions)
    report,score=PrecisionRecall(y_test,predictions)
    messagebox.showinfo('Training complete','Training complete')

    window=Tk()
    window.title("Mall recognition system")
    window.geometry('600x700')
    window.configure(background='palegreen')


    lbl1=Label(window,text
              ="Classification Report",fg="navy",bg="palegreen",font=("Times New Roman", 20))
    lbl1.pack()

    lbl2=Label(window,text=report,fg="navy",bg="palegreen",font=("Arial", 10))
    lbl2.pack()

    lbl3=Label(window,text="Confusion Matrix",fg="navy",bg="palegreen",font=("Times New Roman", 20))
    lbl3.pack()

    lbl4=Label(window,text=conMat,fg="navy",bg="palegreen",font=("Arial", 10))
    lbl4.pack()

    lbl5=Label(window,text="Accuracy Score",fg="navy",bg="palegreen",font=("Times New Roman", 20))
    lbl5.pack()

    lbl6=Label(window,text=score,fg="navy",bg="palegreen",font=("Arial", 10))
    lbl6.pack()

    from tkinter import filedialog as fd
    filename=fd.askopenfilename()

    #classify new image
    basewidth = 100
    hsize = 100
    image = PIL.Image.open(filename)
    image = image.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    image = image.convert("L")
    data = image.getdata()
    data1 = np.matrix(data)
    predIm = pcaIm(X_train,data1)
    predictions=svm(X_train_pca,y_train,predIm)

    if predictions == 1:
        text = "The photo is Aeon"
    elif predictions == 2:
        text ="The photo is Melaka Mall"
    else:
        text="The photo is Mydin"
    messagebox.showinfo('Classifying result',text)

    
def main():
    window = Tk()
    window.title("Mall Recognition System")
    window.geometry('600x400')
    window.configure(background='palegreen')

    lbl=Label(window,text
              ="Mall Recognition System",fg="navy",bg="palegreen",font=("Times New Roman", 30))
    lbl.pack()

    lbl1=Label(window,text="By Jeanny, Vellnica, Erin, Jonathon \n\n",fg="navy",bg="palegreen",font=("Arial", 10))
    lbl1.pack()

    lbl2=Label(window,text="What does this system do?",fg="navy", bg="palegreen", font=("Arial Bold", 15))
    lbl2.pack()

    lbl3=Label(window,text="This system is used to train a classifier to classify\n" 
                "three different malls which are Aeon Mall, Melaka Mall and Mydin\n\n",fg="navy",bg="palegreen",font=("Arial", 10))
    lbl3.pack()


    lbl5=Label(window,text="To start the system training, click the Train button\n\n",fg="navy", bg="palegreen", font=("Arial", 10))
    lbl5.pack()

    lbl4=Label(window,text="Train the system",fg="navy", bg="palegreen", font=("Arial", 10))
    lbl4.pack()

    btn1 = Button(window, text="Train",command=clicked1)
    btn1.pack()

    window.quit()
    window.mainloop()

main()
    
    
