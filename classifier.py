import itertools
import csv
from scipy.sparse import csr_matrix
from sklearn.svm import SVC

__author__ = 'carlos.ginestra'

from training_factory import TrainingFactory
from testing_factory import TestingFactory
from training_factory import CATEGORIES

from visual import Visual

from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from matplotlib import pyplot as plt
import numpy

class ClassifierFactory():

    @staticmethod
    def go():

        input = TrainingFactory.build_sparse_matrix_input(limit=10000)
        targets = TrainingFactory.build_sparse_matrix_target(limit=10000)

        input_train, input_test, target_train, target_test = train_test_split(input, targets, test_size=0.1)

        classif = OneVsRestClassifier(SVC(kernel='rbf', tol=0.001, probability=True))
        classif.fit(input_train, target_train)

        output_targets = classif.predict_proba(input_test)
        print ClassifierFactory.output_function(output_targets)
        print ClassifierFactory.output_function(target_test.todense())

        print log_loss(target_test, output_targets)
        print

    @staticmethod
    def output_function(list_of_vectors):
        predicted = []
        for vector in list_of_vectors:
            predicted.append(vector.argmax())
        return predicted


    @staticmethod
    def go_by_category(category):
        input = TrainingFactory.build_sparse_matrix_target(limit=10000)
        targets = TrainingFactory.build_target_vector_by_category(category,limit=10000)

        input_train, input_test, target_train, target_test = train_test_split(input, targets, test_size=0.1)

        classif = SVC(kernel='rbf', tol=0.001, probability=True)
        classif.fit(input_train, target_train)

        output_targets = classif.predict(input_test)
        print output_targets
        print target_test
        print


    @staticmethod
    def go_by_category_2(category):
        input, targets, scaler = TrainingFactory.get_training_data_by_category(category,10000)
        input_train, input_test, target_train, target_test = train_test_split(input, targets, test_size=0.1)

        test_data_sparse = TestingFactory.get_test_data(limit=10000)
        test_data_scaled = scaler.transform(test_data_sparse)
        test_data = csr_matrix(test_data_scaled)

        classif = SVC(kernel='rbf',C=0.1, tol=0.001, probability=True)
        classif.fit(input_train, target_train)

        output_targets_proba = classif.predict_proba(input_test)

        outputs_predicted_proba = [item[1] for item in output_targets_proba]
        output_targets = classif.predict(input_test)

        # print output_targets.tolist()
        # print outputs_predicted_proba
        # print target_test

        print log_loss(target_test, output_targets)
        accuracy = accuracy_score(target_test, output_targets)
        ##print accuracy
        ##print confusion_matrix(target_test, output_targets)


        testing_output = classif.predict_proba(test_data)
        testing_output_proba = [item[1] for item in testing_output]
        ## print "\n\n\nproba- ", testing_output_proba, "\n\n\n"
       
        ###
        return accuracy, output_targets, testing_output_proba




if __name__ == "__main__":

    test_results = []
    accuracies= []

    Visual.visualize()
    
    for category in CATEGORIES:
        print "\n\n\n\n---------", category
        acc, targets, testing_out = ClassifierFactory.go_by_category_2(category)
        accuracies.append(acc)
        test_results.append(testing_out)

        ## print "\n\n\nproba- ", testing_out, "\n"

    test_to_k = numpy.column_stack(tuple(test_results))
    print test_to_k
    ## print test_results
    
    output = test_to_k.tolist()
    print len(test_to_k),"---------------",len(output)
    predictions_file = open("submissionDT.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id",'ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT',
                               'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION',
                               'FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT',
                               'LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES',
                               'PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY',
                               'SECONDARY CODES','SEX OFFENSES FORCIBLE','SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY',
                               'SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS',
                               'WEAPON LAWS'])
    for x in range(len(output)):
        output[x].insert(0, x)
        open_file_object.writerow(output[x])
    predictions_file.close()

    x = range(len(accuracies))
    plt.plot(x,accuracies)
    plt.xticks(x,CATEGORIES,rotation='vertical')
    plt.show()
