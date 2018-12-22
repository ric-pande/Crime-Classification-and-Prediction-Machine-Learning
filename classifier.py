import itertools
import csv
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression as lr
from sklearn.naive_bayes import MultinomialNB
from training_factory import TrainingFactory
from testing_factory import TestingFactory
from training_factory import CATEGORIES
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,precision_score
from expolatory_graphs import visualize
from matplotlib import pyplot as plt
import numpy
import expolatory_graphs
import Queue
import time
import visual
#from ui import *
from Tkinter import *
import webbrowser
import threading

import tkMessageBox
import tkFont
from PIL import ImageTk,Image
import Tkinter


#import Image 
#import ImageTk

class ClassifierFactory():


    @staticmethod
    def go_by_category_2(category,num):
        #print category
        #return
        input, targets, scaler = TrainingFactory.get_training_data_by_category(category,10000)
        
        # Split arrays into random train and test subsets
        input_train, input_test, target_train, target_test = train_test_split(input, targets, test_size=0.1)

        test_data_sparse = TestingFactory.get_test_data(limit=10000)
        test_data_scaled = scaler.transform(test_data_sparse)
        test_data = csr_matrix(test_data_scaled)
        if(num==1):
            classif = SVC(kernel='rbf',C=0.1, tol=0.001, probability=True)
        elif(num==0):
            classif= rfc(n_estimators=500, oob_score=True)
        elif(num==2):
            classif = MultinomialNB()
        elif(num==3):
            classif = tree.DecisionTreeClassifier(max_depth=1000)
        else:
            classif = SVC(kernel='linear',C=0.1, tol=0.001, probability=True)
        #classif.fit(input_train, target_train)
        classif.fit(input_train, target_train)

        #output_targets_proba = classif.predict_proba(input_test)

        #outputs_predicted_proba = [item[1] for item in output_targets_proba]
        output_targets = classif.predict(input_test)

        # print output_targets.tolist()
        # print outputs_predicted_proba
        # print target_test

        print "log loss: ",log_loss(target_test, output_targets)
        accuracy = accuracy_score(target_test, output_targets)
        print "accuracy: ",accuracy

        cm= confusion_matrix(target_test, output_targets)
        print "Confusion matrix :",cm,"\n"
        #print "prediction score"
        #print precision_score(target_test,output_targets,average="macro")

        testing_output = classif.predict_proba(test_data)
        testing_output_proba = [item[1] for item in testing_output]
       
        ###
        return accuracy, output_targets, testing_output_proba




#if __name__ == "__main__":
    #app=App()
    


    

    # expolatory_graphs.visualize()
class MyThread(threading.Thread):
    def run(self):
        time.sleep(5)
        print "fg"
        return

class ThreadedTask(threading.Thread):
    def __init__(self, num):
        threading.Thread.__init__(self)
        self.num=num
        #print num
        #self.queue = queue
        
    def run(self):
        test_results = []
        accuracies= []
        for category in CATEGORIES:
            print "Learning: ", category,"\n"
            acc, targets, testing_out = ClassifierFactory.go_by_category_2(category,self.num)
            accuracies.append(acc)
            test_results.append(testing_out)

            ## print "\n\n\nproba- ", testing_out, "\n"
    
        test_to_k = numpy.column_stack(tuple(test_results))
        ##print test_to_k
        ## print test_results
        
        output = test_to_k.tolist()
        print "Writing ",len(test_to_k),"predictions"
        predictions_file = open("op.csv", "wb")
        open_file_object = csv.writer(predictions_file)
        open_file_object.writerow(["Id",'ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT',
                                   'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION',
                                   'FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT',
                                   'LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES',
                                   'PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY',
                                   'SECONDARY CODES','SEX OFFENSES FORCIBLE','SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY',
                                   'SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS',
                                   'WEAPON LAWS', 'PREDICTED CATEGORY'])
        for x in range(len(output)):
            maxp=output[x].index(max(output[x]))
            output[x].insert(0, x)
            output[x].append(CATEGORIES[maxp])
            open_file_object.writerow(output[x])
        predictions_file.close()

        x = range(len(accuracies))
        #plt.plot(x,accuracies)
        #plt.ylabel('Accuracies')
        #plt.xticks(x,CATEGORIES,rotation='vertical')
        #plt.show()
        
        while True :
            row= int(raw_input("Enter id (-1 to terminate) and -2 for overall : "))
            if (row < 0 and row!= -2) or row >= len(output):
                break
            if (row == -2):
                plt.plot(x,accuracies)
                plt.ylabel('Accuracies')
                plt.xticks(x,CATEGORIES,rotation='vertical')
                plt.show()
                break
            plt.plot(x,output[row][1:-1:])
            plt.xticks(x,CATEGORIES,rotation='vertical')
            plt.show()

class App:
    opt=["MODIFIED","SVM","Naive Bayes","Decision tree","Random Forest"]
    def run_algo(self,sel):
        i=App.opt.index(sel)
        if i<0:
            i=0
        #App.startc(i)
        #self.queue = Queue.Queue()
        ThreadedTask(i).start()
        #t = threading.Thread(target=App.startc, args=(i,))
        #MyThread().start()
        #self.master.after(100, self.process_queue)
        #t.join()

    def process_queue(self):
        try:
            msg = self.queue.get(0)
            # Show result of the task if needed
            self.prog_bar.stop()
        except Queue.Empty:
            self.master.after(100, self.process_queue)
        
    def __init__(self, master):
        self.master=master
        master.title("CRIME CLASSIFICATION AND PREDICTION")
        frame = Frame(master)
        frame.pack(padx=10, pady=10)
        self.btn2 = Button(frame,
                             text="EXPOLATORY_GRAPHS",bg="yellow",fg="black",
                             command= lambda: expolatory_graphs.visualize())
        #self.btn2.pack(fill=X,padx=10, pady=10)
        self.btn2.grid(row=0,column=0)
        self.btn4 = Button(frame,
                            text="VISUALIZE_DATA",bg="yellow",fg="black",
                            command= lambda: visual.Visual.visualize())
        #self.btn4.pack(fill=X,padx=10, pady=10)
        self.btn4.grid(row=0,column=1)
        self.btn3 = Button(frame,
                             text="Interactive Map",bg="purple",fg="black",
                             command= lambda: webbrowser.open("file:///C:/Users/fugga/Desktop/MAJOR_PROJECT/OMG/sf-crime-map-master/index.html"))
        #self.btn3.pack(fill=X,padx=10, pady=10)
        self.btn3.grid(row=0,column=3)
        var=StringVar(frame)
        var.set("Choose Algorithm")
        option=OptionMenu(frame,var,*self.opt,command=self.run_algo)
        #option.pack(fill=X,padx=10, pady=10)
        option.grid(row=1,column=1)
        self.btn1 = Button(frame, 
                             text="QUIT", bg="red",fg="black",
                             command=self.quit)
        #self.btn1.pack(fill=X,padx=10, pady=10)
        self.btn1.grid(row=2,column=1)
    def quit(self):
        root.destroy()
       
root = Tk()
#root.geometry("1111x675+300+300")
im = Image.open('C:/Users/fugga/Desktop/MAJOR_PROJECT/OMG/crimescene.jpg')
#tkimage = ImageTk.PhotoImage(im)
#myvar=Tkinter.Label(root,image = tkimage)
#myvar.place(x=0, y=0, relwidth=1, relheight=1)
#i=Image.open("C:/Users/fugga/Desktop/MAJOR_PROJECT/OMG/crimescene.jpg")
background_image=ImageTk.PhotoImage(im)
background_label = Tkinter.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
App(root)
root.mainloop()
