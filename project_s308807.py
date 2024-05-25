import numpy as np
import matplotlib.pyplot as plt

from utils import *

from pca import apply_pca
from lda import compute_lda_matrix,compute_SB,compute_SW,evaluate
from gau import loglikelihood,logpdf_GAU_ND,compute_mu_C,Gau_MVG_ML_estimates, compute_log_likelihood_Gau, compute_logPosterior

GENUINE = 1
FAKE = 0


def show_hists(genuine,fake):
    # For each feature
    for feature in range(6):
        # Create the histogram for the selected feature
        plt.figure()
        plt.hist(genuine[feature], density=True, alpha = 0.5, label= "Genuine",bins= 30)
        plt.hist(fake[feature], density=True, alpha = 0.5, label= "Fake",bins= 30)

        plt.legend(loc = "upper right")
        plt.xlabel("Feature " + str(feature))
        plt.title("Feature %d" % (feature + 1))
        plt.savefig('./histograms/hist_%d.png' % feature)
    #plt.show()

def show_hists_v2(data, labels, folder):
    # For each feature
    for feature in range(6):
        # Create the histogram for the selected feature
        plt.figure()
        plt.hist(data[:,labels == GENUINE][feature], density=True, alpha = 0.5, label= "Genuine",bins= 30)
        plt.hist(data[:,labels == FAKE][feature], density=True, alpha = 0.5, label= "Fake",bins= 30)

        plt.legend(loc = "upper right")
        plt.xlabel("Feature " + str(feature))
        plt.title("Feature %d" % (feature + 1))
        plt.savefig(folder + "/hist" + str(feature) + ".png")
    #plt.show()

def show_scatter(genuine,fake):
    # For each feature
    for feature1 in range(6):
        for feature2 in range(6):
            if feature1 == feature2:
                continue
            # Create the scatter plot
            plt.figure()
            plt.scatter(genuine[feature1],genuine[feature2],alpha=0.7 ,label= "Genuine")
            plt.scatter(fake[feature1],fake[feature2], alpha=0.7, label= "Fake")

            plt.legend(loc = "upper right")
            plt.xlabel("Feature " + str(feature1))
            plt.ylabel("Feature " + str(feature2))
            plt.title("Feature %d vs Feature %d" % (feature1,feature2))
            plt.savefig('./scatter/scatter_%d_%d.png' % (feature1, feature2))
        #plt.show()


    return



def find_best_threshold(train_data,initial_threshold,LTR):
    threshold_list = np.linspace(initial_threshold-(initial_threshold/2.0),initial_threshold+(initial_threshold/2.0),num=300)
    best_t = initial_threshold
    PVAL = np.zeros(shape=LTR.shape, dtype=np.int32)
    PVAL[train_data[0] >= initial_threshold] = GENUINE
    PVAL[train_data[0] < initial_threshold] = FAKE

    errors = PVAL == LTR
    errors_count = 0

    for e in errors:
        if not e:
            errors_count += 1
    best_accuracy = 100.0*(errors.shape[0]-errors_count)/float(errors.shape[0])
    
    for threshold in threshold_list:
        
        PVAL = np.zeros(shape=LTR.shape, dtype=np.int32)
        PVAL[train_data[0] >= threshold] = GENUINE
        PVAL[train_data[0] < threshold] = FAKE

        errors = PVAL == LTR
        errors_count = 0

        for e in errors:
             if not e:
                  errors_count += 1
        
        if 100.0*(errors.shape[0]-errors_count)/float(errors.shape[0]) > best_accuracy:
            best_accuracy = 100.0*(errors.shape[0]-errors_count)/float(errors.shape[0])
            best_t = threshold
        #print("With a threshold of: ",threshold, " Accuracy: ",100.0*(errors.shape[0]-errors_count)/float(errors.shape[0]),"%")
    return best_t, 100.0 - best_accuracy

def main():
    # Open the file
    f = open("trainData.txt","r")

    # Parse the file
    data,labels = parse_file(f)

    # Labs until lab4
    #until_lab4(data,labels)

    # Lab5
    print(">>> START: Lab 5 with all the feature\n")
    lab5(data,labels)
    print(">>> END: Lab 5 with all the feature\n")

    print(">>> START: Lab 5 with only the first 4 features\n")
    lab5(data[0:4,:],labels, mode="first_4")
    print(">>> END: Lab 5 with only the first 4 features\n")

    print(">>> START: Lab 5 with only the first 2 features\n")
    lab5(data[0:2,:],labels, mode="first_2")
    print(">>> END: Lab 5 with only the first 2 features\n")

    print(">>> START: Lab 5 with only the features 3-4\n")
    lab5(data[2:4,:],labels, mode="last_2")
    print(">>> END: Lab 5 with only features 3-4\n")
    
    print(">>> START: Lab 5 with PCA\n")
    lab5(apply_pca(data,m = 4), labels, mode="pca")
    print(">>> END: Lab 5 with PCA\n")

    print(">>> START: Lab 7\n")
    lab7(data, labels)
    print(">>> END: Lab 7\n")

def lab7(data,labels):
    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)
    llr = np.load("./backups/LLR_MVG.npy")

    from confusion_matrix import (compute_optimal_Bayes_binary_llr, 
                                compute_confusion_matrix,
                                compute_empirical_Bayes_risk_binary,
                                compute_empirical_Bayes_risk,
                                compute_minDCF_binary_slow,
                                compute_minDCF_binary_fast,
                                compute_Pfn_Pfp_allThresholds_fast)

    for prior, Cfn, Cfp in [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]:
        print()
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
        
        predictions = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)

        print(compute_confusion_matrix(predictions, LVAL))
    
        print('DCF (non-normalized): %.3f' % (compute_empirical_Bayes_risk_binary(
            predictions, LVAL, prior, Cfn, Cfp, normalize=False)))
        print('DCF (non-normalized, multiclass code): %.3f' % (compute_empirical_Bayes_risk(
            predictions,
            LVAL,
            np.array([1-prior, prior]), # Class 1 is the second element for multiclass
            np.array([[0, Cfn], [Cfp, 0]]),
            normalize=False)))
        print('DCF (normalized): %.3f' % (compute_empirical_Bayes_risk_binary(
            predictions, LVAL, prior, Cfn, Cfp)))
        print('DCF (normalized, multiclass code): %.3f' % (compute_empirical_Bayes_risk(
            predictions,
            LVAL,
            np.array([1-prior, prior]), # Class 1 is the second element for multiclass
            np.array([[0, Cfn], [Cfp, 0]]))))
        
        minDCF, minDCFThreshold = compute_minDCF_binary_fast(llr, LVAL, prior, Cfn, Cfp, returnThreshold=True)
        print('MinDCF (normalized, fast): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))

    # ROC plot - uncomment the commented lines to see the plot
    Pfn, Pfp, _ = compute_Pfn_Pfp_allThresholds_fast(llr, labels)
    plt.figure(0)
    plt.plot(Pfp, 1-Pfn)
    plt.show()

    # Bayes error plot
    effPriorLogOdds = np.linspace(-3, 3, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
        predictions = compute_optimal_Bayes_binary_llr(llr, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(predictions, LVAL, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(llr, LVAL, effPrior, 1.0, 1.0))
    plt.figure(1)
    plt.plot(effPriorLogOdds, actDCF, label='actDCF eps 0.001', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='DCF eps 0.001', color='b')
    plt.ylim([0, 1.1])
    plt.show()
            
#    commedia_llr_binary = np.load('../Data/commedia_llr_infpar_eps1.npy')
#    commedia_labels_binary = np.load('../Data/commedia_labels_infpar_eps1.npy')
#
#    actDCF = []
#    minDCF = []
#    for effPrior in effPriors:
#        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
#        commedia_predictions_binary = compute_optimal_Bayes_binary_llr(commedia_llr_binary, effPrior, 1.0, 1.0)
#        actDCF.append(compute_empirical_Bayes_risk_binary(commedia_predictions_binary, commedia_labels_binary, effPrior, 1.0, 1.0))
#        minDCF.append(compute_minDCF_binary_fast(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0))
#
#    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='actDCF eps 1.0', color='y')
#    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='DCF eps 1.0', color='c')
#    matplotlib.pyplot.ylim([0, 1.1])
#
#    matplotlib.pyplot.legend()
#    matplotlib.pyplot.show()
#
#
    


def lab5(data, labels, mode = "default"):

    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)

    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)

    for lab in [0,1]:
        #print('MVG - Class', lab)
        #print(hParams_MVG[lab][0])
        #print(hParams_MVG[lab][1])
        print()



    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])

    if mode == "default":
        np.save("./backups/LLR_MVG.npy",LLR)

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("MVG - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))     
    print()

    from gau import Gau_Tied_ML_estimates

    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)
    for lab in [0,1]:
        #print('Tied Gaussian - Class', lab)
        #print(hParams_Tied[lab][0])
        #print(hParams_Tied[lab][1])
        print()

    LLR = logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1]) - logpdf_GAU_ND(DVAL, hParams_Tied[0][0], hParams_Tied[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("Tied - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))     

    print()

    from gau import Gau_Naive_ML_estimates

    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)
    for lab in [0,1]:
        #print('Naive Bayes Gaussian - Class', lab)
        #print(hParams_Naive[lab][0])
        #print(hParams_Naive[lab][1])
        print()
    
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_Naive)
    S_logPost = compute_logPosterior(S_logLikelihood, np.ones(2)/2.)
    PVAL = S_logPost.argmax(0)
    print("Naive Bayes Gaussian - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
    print()


    C = hParams_MVG[0][1]
    Corr_0 = C / ( v_col(C.diagonal()**0.5) * v_row(C.diagonal()**0.5) )
    print("Correlation Matrix for class 0:\n",Corr_0)

    C = hParams_MVG[1][1]
    Corr_1 = C / ( v_col(C.diagonal()**0.5) * v_row(C.diagonal()**0.5) )
    print("Correlation Matrix for class 1:\n",Corr_1)





def until_lab4(data,labels):

    print("Data shape: ",data.shape)
    print("Label shape: ",labels.shape)

    # Split by class
    genuine = data[:,labels == GENUINE]
    fake = data[:,labels == FAKE]

    # Compute the mean for the genuine class and center the data
    genuine_centered, mean_genuine = center_data(genuine)

    print("Mean genuine class: \n", mean_genuine)

    # Compute the mean for the fake class and center the data
    fake_centered, mean_fake = center_data(fake)

    print("Mean fake class: \n", mean_fake)

    # Get the variance and standard daviation for genuine class
    var_genuine,std_genuine = compute_var_std(genuine_centered)

    print("Variance for genuine: \n",var_genuine,"\nStandard deviation for genuine: \n",std_genuine)

    # Get the variance and standard daviation for fake class
    var_fake,std_fake = compute_var_std(fake_centered)

    print("Variance for fake: \n",var_fake,"\nStandard deviation for fake: \n",std_fake)

    #show_hists(genuine,fake)
    
    #show_scatter(genuine,fake)

    #show_hists(genuine_centered,fake_centered)

    #show_scatter(genuine_centered,fake_centered)


    #Lab 3

    # Center data
    centered_data, mean = center_data(data)

    # Compute the covariance matrix
    covariance_matrix = (centered_data @ centered_data.T)/float(data.shape[1])
    print("Covariance matrix:\n",covariance_matrix)

    # Compute eigenvalue and eigenvectors
    eigenvalues,eigenvectors = np.linalg.eigh(covariance_matrix)
    print("Eigenvalues:\n",eigenvalues) # They are all > 0
    # Sort the eigenvectors in descending order
    P = eigenvectors[:,::-1]

    # Compute the projection of the entire dataset in the new space
    y = np.dot(P.T,data)

    # Plot the histograms
    #show_hists_v2(y, labels,"./PCA/Hists")
    
    # Compute the within class covariance matrix 
    Sw = compute_SW(genuine,genuine_centered,fake,fake_centered,data)
    print("Sw:\n", Sw)

    # Compute the between class covariance matrix 
    Sb = compute_SB(genuine,mean_genuine,fake,mean_fake,mean,data)
    print("Sb:\n", Sb)

    # Compute Singular Value Decomposition for Sw
    U,s,_ = np.linalg.svd(Sw)

    # P1 can be calculated as follows
    P1 = np.dot(U*v_row(1.0/(s**0.5)),U.T)

    # Compute the transformed between class covariance
    Sbt = P1 @ Sb @ P1.T
    print("Sbt:\n",Sbt)
    
    # Obtain the eigenvalues and eigenvectors of Sbt
    lda_eigenvalues,lda_eigenvectors = np.linalg.eigh(Sbt)

    # we have only one class so apply lda with 1 dimension
    m = 1

    # Sort the eigenvectors and take the m highest
    P2 = lda_eigenvectors[:,::-1][:, 0:m]
    print(lda_eigenvectors)

    # W is the lda matrix
    W = P1.T @ P2
    print("W:\n",W)

    # The single dimension data for lda
    y_lda = W.T @ data

    # Histogram of lda, the classes overlap a bit
    plt.figure()
    plt.hist(y_lda[:,labels == GENUINE][0], density=True, alpha = 0.5, label= "Genuine",bins= 30)
    plt.hist(y_lda[:,labels == FAKE][0], density=True, alpha = 0.5, label= "Fake",bins= 30)

    plt.legend(loc = "upper right")
    plt.xlabel("Feature " + str(0))
    plt.title("Feature %d" % (1))
    plt.savefig("LDA/Hists/hist" + str(0) + ".png")

    #plt.show()

    # Split in 2/3 for training and 1/3 for val
    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)

    # Split the train dataset and compute the train mean and the mean per class
    genuine_train = DTR[:,LTR == GENUINE]
    fake_train = DTR[:,LTR == FAKE]

    genuine_centered_train, mean_genuine_train = center_data(genuine_train)
    fake_centered_train, mean_fake_train = center_data(fake_train)
    DTR_centered, mean_train = center_data(DTR)

    # Compute the within class covariance
    Sw_train = compute_SW(genuine_train,genuine_centered_train,fake_train,fake_centered_train,DTR)
    print("Sw train:\n",Sw_train)
    
    # Compute the between class covariance matrix
    Sb_train = compute_SB(genuine_train,mean_genuine_train,fake_train,mean_fake_train,mean_train,DTR)
    print("Sb train:\n",Sb_train)

    # Compute lda matrix and ptoject the training and val data
    lda_matrix_train = compute_lda_matrix(Sw_train,Sb_train)
    print("Lda matrix train:\n",lda_matrix_train)

    y_train = lda_matrix_train.T @ DTR
    y_val = lda_matrix_train.T @ DVAL

    
    plt.figure()
    plt.hist(y_train[:,LTR == GENUINE][0], density=True, alpha = 0.5, label= "Genuine",bins= 30)
    plt.hist(y_train[:,LTR == FAKE][0], density=True, alpha = 0.5, label= "Fake",bins= 30)

    plt.legend(loc = "upper right")
    plt.title("Train")

    plt.figure()
    plt.hist(y_val[:,LVAL == GENUINE][0], density=True, alpha = 0.5, label= "Genuine",bins= 30)
    plt.hist(y_val[:,LVAL == FAKE][0], density=True, alpha = 0.5, label= "Fake",bins= 30)

    plt.legend(loc = "upper right")
    plt.title("Val")
    #plt.show()
    
    # Compute the treshold as the average of projected means
    threshold = (y_train[0, LTR==GENUINE].mean() + y_train[0, LTR==FAKE].mean()) / 2.0
    
    # Apply the threshold to the classification for the val dataset to test the accuracy
    accuracy = evaluate(y_val,LVAL,threshold)
   
    print("With an initial threshold of: ",threshold, " Error rate on val: ",100.0 - accuracy,"%")
    
    
    
    # Try several values for the threshold in the training data
    threshold, error_rate = find_best_threshold(y_train,threshold,LTR)
    print("With a threshold of: ",threshold, " Error rate on train: ",error_rate,"%")

    # Evaluate the final threshold usign validation set
    accuracy = evaluate(y_val,LVAL,threshold)
    
    print("With a threshold of: ",threshold, " Error rate on val: ",100.0 - accuracy,"%")
    best_accuracy = 0.0
    best_m = 0
    best_threshold = 0.0
    for m in range(2,6):

        y_pca,val_pca = apply_pca(DTR,DVAL,m)

        # Split the train dataset and compute the train mean and the mean per class
        genuine_train = y_pca[:,LTR == GENUINE]
        fake_train = y_pca[:,LTR == FAKE]

        genuine_centered_train, mean_genuine_train = center_data(genuine_train)
        fake_centered_train, mean_fake_train = center_data(fake_train)
        DTR_centered, mean_train = center_data(y_pca)

        # Compute the within class covariance
        Sw_train = compute_SW(genuine_train,genuine_centered_train,fake_train,fake_centered_train,y_pca)

        # Compute the between class covariance matrix
        Sb_train = compute_SB(genuine_train,mean_genuine_train,fake_train,mean_fake_train,mean_train,y_pca)

        # Compute lda matrix and ptoject the training and val data
        lda_matrix_train = compute_lda_matrix(Sw_train,Sb_train)

        y_train = lda_matrix_train.T @ y_pca
        y_val = lda_matrix_train.T @ val_pca

        # Compute the treshold as the average of projected means
        threshold = (y_train[0, LTR==GENUINE].mean() + y_train[0, LTR==FAKE].mean()) / 2.0
    
        # Apply the threshold to the classification for the train dataset to test the accuracy
        
        accuracy = evaluate(y_train,LTR,threshold)
        print("m: ",m, " Error rate on train: ",100.0 - accuracy,"%")
        
        if accuracy > best_accuracy:
            print(accuracy,best_accuracy)
            best_accuracy = accuracy
            best_m = m
            best_threshold = threshold
            accuracy_val = evaluate(y_val,LVAL,threshold)
            
    print("Select m: ",best_m, " Error rate on val: ", 100.0 - accuracy_val, "%" )
    
    # Lab 4

    # Compute the ML estimates for each feature
    
    for idx,feature in enumerate(data):
        feature = v_row(feature)
        
        feature_genuine = feature[:,labels == GENUINE]
        feature_fake = feature[:,labels == FAKE]

        centered_feature_genuine, mean_feature_genuine = center_data(feature_genuine)
        covariance_matrix_genuine = (centered_feature_genuine @ centered_feature_genuine.T)/float(feature_genuine.shape[1])

        centered_feature_fake, mean_feature_fake = center_data(feature_fake)
        covariance_matrix_fake = (centered_feature_fake @ centered_feature_fake.T)/float(feature_fake.shape[1])

        print("Mean for feature %d and class Genuine: " % idx,mean_feature_genuine)
        print("Mean for feature %d and class Fake: " % idx,mean_feature_fake)
        print("Variance for feature %d and class Genuine:" % idx,covariance_matrix_genuine)
        print("Variance for feature %d and class Fake: " % idx,covariance_matrix_fake)



        #print(v_col(feature).shape)
        plt.figure()
        plt.hist(feature_genuine[0], density=True, alpha = 0.5, label="Genuine", bins= 30)
        plt.hist(feature_fake[0],density=True,alpha = 0.5, label="Fake", bins=30)

        x_genuine = np.linspace(feature_genuine[0].min(), feature_genuine[0].max(), 1000)
        
        plt.plot(x_genuine.ravel(), np.exp(logpdf_GAU_ND(v_row(x_genuine), mean_feature_genuine, covariance_matrix_genuine)),label = "Genuine")

        x_fake = np.linspace(feature_fake[0].min(), feature_fake[0].max(), 1000)

        plt.title("Feature %d" % (idx + 1))
        
        plt.plot(x_fake.ravel(), np.exp(logpdf_GAU_ND(v_row(x_fake), mean_feature_fake, covariance_matrix_fake)),label = "Fake")

        plt.legend(loc = "upper right")
        plt.savefig("GAU/hists/hist" + str(idx) + ".png")
        #plt.show()


    return







if __name__ == "__main__":
    main()