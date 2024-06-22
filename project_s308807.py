import numpy as np
import matplotlib.pyplot as plt

from utils import *

from pca import apply_pca, compute_pca, apply_pca_sol
from lda import compute_lda_matrix,compute_SB,compute_SW,evaluate, compute_lda_JointDiag, apply_lda
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
 #   until_lab4(data,labels)
 #   print()
 #   # Lab5
 #   print(">>> START: Lab 5 with all the feature\n")
 #   lab5(data,labels)
 #   print(">>> END: Lab 5 with all the feature\n")
#
 #   print(">>> START: Lab 5 with only the first 4 features\n")
 #   lab5(data[0:4,:],labels, mode="first_4")
 #   print(">>> END: Lab 5 with only the first 4 features\n")
#
 #   print(">>> START: Lab 5 with only the first 2 features\n")
 #   lab5(data[0:2,:],labels, mode="first_2")
 #   print(">>> END: Lab 5 with only the first 2 features\n")
#
 #   print(">>> START: Lab 5 with only the features 3-4\n")
 #   lab5(data[2:4,:],labels, mode="last_2")
 #   print(">>> END: Lab 5 with only features 3-4\n")
 #   
 #   for m in range(1,6):
 #       print(">>> START: Lab 5 with PCA m = "+ str(m) +"\n")
 #       lab5(data, labels, mode="pca_m"+str(m), m = m)
 #       print(">>> END: Lab 5 with PCA m = " + str(m) + "\n")
#
 #   prior_cost_list = [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]
#
 #   for prior, Cfn, Cfp in prior_cost_list:
 #       effective_prior = (prior*Cfn)/(prior*Cfn + (1-prior)*Cfp)
 #       print(f'Effective prior for configuration {prior} {Cfn} {Cfp} = ', effective_prior)
#
 #   # Lab7
#
 #   best_m_MVG = 0
 #   best_DCF_MVG = 99999999999999.0
#
 #   
 #   best_m_Tied = 0
 #   best_DCF_Tied = 99999999999999.0
#
 #   
 #   best_m_Naive = 0
 #   best_DCF_Naive = 99999999999999.0
 #   
#
 #   print(">>> START: Lab 7 MVG\n")
 #   lab7(data, labels)
 #   print(">>> END: Lab 7 MVG\n")
 #   
 #   print(">>> START: Lab 7 Tied\n")
 #   lab7(data, labels, model="Tied")
 #   print(">>> END: Lab 7 Tied\n")
#
 #   print(">>> START: Lab 7 Naive\n")
 #   lab7(data, labels, model="Naive")
 #   print(">>> END: Lab 7 Naive\n")
#
 #   for m in range(1,6):
#
 #       print(">>> START: Lab 7 PCA MVG\n", "m = ", m)
#
 #       dcf,actDCF = lab7(data, labels, mode="pca_m"+str(m),m=m)
#
 #       if(dcf < best_DCF_MVG):
 #           best_m_MVG = m
 #           best_DCF_MVG = dcf
 #           best_act_MVG = actDCF
#
 #       print(">>> END: Lab 7 PCA MVG\n")
#
 #       print(">>> START: Lab 7 PCA Tied\n" "m = ", m)
 #       dcf,actDCF = lab7(data, labels, model="Tied", mode="pca_m"+str(m), m=m)
 #       if(dcf < best_DCF_Tied):
 #           best_m_Tied = m
 #           best_DCF_Tied = dcf
 #           best_act_Tied = actDCF
 #       print(">>> END: Lab 7 PCA Tied\n")
#
 #       print(">>> START: Lab 7 PCA Naive\n" "m = ", m)
 #       dcf,actDCF = lab7(data, labels, model="Naive", mode="pca_m"+str(m), m=m)
 #       if(dcf < best_DCF_Naive):
 #           best_m_Naive = m
 #           best_DCF_Naive = dcf
 #           best_act_Naive = actDCF
 #       print(">>> END: Lab 7 PCA Naive\n")
#
 #       #print("The best model for prior 0.1 with PCA m = ", best_m, ", model = ", best_model, "minDCF = ", best_DCF)
#
 #   print(f'MVG with PCA, best m = {best_m_MVG}, DCF = {best_DCF_MVG}, Calibration error: {100.0*(best_act_MVG-best_DCF_MVG)/best_DCF_MVG}')
 #   print(f'Tied with PCA, best m = {best_m_Tied}, DCF = {best_DCF_Tied}, Calibration error: {100.0*(best_act_Tied-best_DCF_Tied)/best_DCF_Tied}')
 #   print(f'Naive with PCA, best m = {best_m_Naive}, DCF = {best_DCF_Naive}, Calibration error: {100.0*(best_act_Naive-best_DCF_Naive)/best_DCF_Naive}')


    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(data, labels)

#    print(">>> START: Lab 8 \n")
#    lab8(DTR,LTR,DVAL,LVAL)
#    print(">>> END: Lab 8 \n")
#
#    print(">>> START: Lab 8 only few samples\n")
#    lab8(DTR[:,::50],LTR[::50],DVAL,LVAL, mode='few_samples')
#    print(">>> END: Lab 8 only few samples\n")

#    print(">>> START: Lab 9\n")
#    lab9(DTR,LTR,DVAL,LVAL)
#    print(">>> END: Lab 9\n")

    lab10(DTR,LTR,DVAL,LVAL)

    

from gmm import train_GMM_LBG_EM, logpdf_GMM

def lab10(DTR,LTR,DVAL,LVAL):
    for covType in ['full', 'diagonal', 'tied']:
        print(covType)
        for numC in [1,2,4,8,16,32]:
            gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, covType = covType, verbose=False, psiEig = 0.01)
            gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC, covType = covType, verbose=False, psiEig = 0.01)

            SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
            print(numC, covType)
            #print ('minDCF - pT = 0.5: %.4f' % bayesRisk.compute_minDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0))
            #print ('actDCF - pT = 0.5: %.4f' % bayesRisk.compute_actDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0))
            print ('\tnumC = %d: %.4f / %.4f' % (numC, compute_minDCF_binary_fast(SLLR, LVAL, 0.1, 1.0, 1.0), compute_actDCF_binary_fast(SLLR, LVAL, 0.1, 1.0, 1.0)))
            
        print()

from svm import train_dual_SVM_linear, polyKernel,rbfKernel,train_dual_SVM_kernel

def lab9(DTR,LTR,DVAL,LVAL):

    C_values = np.logspace(-5, 0, 11)
    
    minDCF_values = []
    actDCF_values = []

    K = 1.0

    # Linear
    print("\nLinear\n")
    for C in C_values:
        w, b = train_dual_SVM_linear(DTR, LTR, C, K)
        SVAL = (v_row(w) @ DVAL + b).ravel()
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))

        minDCF = compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        actDCF = compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % minDCF)
        print ('actDCF - pT = 0.1: %.4f' % actDCF)
        print ()

        minDCF_values.append(minDCF)
        actDCF_values.append(actDCF)


    plt.figure()
    plt.plot(C_values,minDCF_values, label = "minDCF")
    plt.plot(C_values,actDCF_values, label = "actDCF")
    plt.legend()
    plt.title('minDCF and actDCF in function of C')
    plt.xscale('log',base = 10)
    plt.savefig("SVM/linear_svm.png")


    # Poly
    print("\nPoly\n")
    minDCF_values = []
    actDCF_values = []

    poly_kernel = polyKernel(2,1)

    for C in C_values:
        fScore = train_dual_SVM_kernel(DTR, LTR, C, poly_kernel, 0.0)
        SVAL = fScore(DVAL)
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        
        minDCF = compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        actDCF = compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        
        print ('minDCF - pT = 0.1: %.4f' % minDCF)
        print ('actDCF - pT = 0.1: %.4f' % actDCF)
        print ()

        minDCF_values.append(minDCF)
        actDCF_values.append(actDCF)


    plt.figure()
    plt.plot(C_values,minDCF_values, label = "minDCF")
    plt.plot(C_values,actDCF_values, label = "actDCF")
    plt.legend()
    plt.title('minDCF and actDCF in function of C')
    plt.xscale('log',base = 10)
    plt.savefig("SVM/poly_svm.png")


    # RBF
    print("\nRBF\n")
    eps = 1.0

    C_values = np.logspace(-3, 2, 11)
    
    

    plt.figure()

    from math import exp as e
    for gamma in [e(-4),e(-3),e(-2),e(-1)]:
        minDCF_values = []
        actDCF_values = []
        kernelFunc = rbfKernel(gamma)
        for C in C_values:
            fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps)
            SVAL = fScore(DVAL)
            PVAL = (SVAL > 0) * 1
            err = (PVAL != LVAL).sum() / float(LVAL.size)
            print ('Error rate: %.1f' % (err*100))

            minDCF = compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            actDCF = compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)

            print ('minDCF - pT = 0.1: %.4f' % minDCF)
            print ('actDCF - pT = 0.1: %.4f' % actDCF)
            print ()

            minDCF_values.append(minDCF)
            actDCF_values.append(actDCF)

        plt.plot(C_values,minDCF_values, label = f'minDCF g: {gamma:.2f}')
        plt.plot(C_values,actDCF_values, label = f'actDCF g: {gamma:.2f}')

    plt.legend()
    plt.title('minDCF and actDCF grid search C,gamma')
    plt.xscale('log',base = 10)
    plt.savefig("SVM/rbf_svm.png")

from confusion_matrix import compute_actDCF_binary_fast
from logreg import trainLogRegBinary, trainWeightedLogRegBinary, trainLogRegQuadratic, quadratic_features

def lab8(DTR,LTR,DVAL,LVAL, mode = 'default'):
    
    minDCF_values = []
    actDCF_values = []

    minDCF_values_w = []
    actDCF_values_w = []

    minDCF_values_q = []
    actDCF_values_q = []


    for lam in np.logspace(-4, 2, 13):

        # Linear
        print()
        w, b = trainLogRegBinary(DTR, LTR, lam) # Train model
        sVal = np.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        # Compute LLR-like scores
        sValLLR = sVal - np.log(pEmp / (1-pEmp))
        # Compute optimal decisions for the prior 0.1

        minDCF = compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)
        actDCF = compute_actDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)

        minDCF_values.append(minDCF)
        actDCF_values.append(actDCF)

        print ('minDCF - pT = 0.5: %.4f' % minDCF)
        print ('actDCF - pT = 0.5: %.4f' % actDCF)


        # Weighted
        pT = 0.1
        w, b = trainWeightedLogRegBinary(DTR, LTR, lam, pT = pT) # Train model to print the loss
        sVal = np.dot(w.T, DVAL) + b
        sValLLR = sVal - np.log(pT / (1-pT))

        minDCF_w = compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
        actDCF_w = compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)

        minDCF_values_w.append(minDCF_w)
        actDCF_values_w.append(actDCF_w)

        print ('minDCF - pT = 0.8: %.4f' % minDCF_w)
        print ('actDCF - pT = 0.8: %.4f' % actDCF_w)
        
        # Quadratic
        if mode != 'few_samples':
            print ()
            w, b = trainLogRegQuadratic(DTR, LTR, lam) # Train model
            DVAL_quad = quadratic_features(DVAL)
            print("Quadratic shape val: ",DVAL_quad.shape)
            sVal = np.dot(w.T, DVAL_quad) + b # Compute validation scores
            PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
            err = (PVAL != LVAL).sum() / float(LVAL.size)
            print ('Error rate: %.1f' % (err*100))
            # Compute empirical prior
            pEmp = (LTR == 1).sum() / LTR.size
            # Compute LLR-like scores
            sValLLR = sVal - np.log(pEmp / (1-pEmp))
            # Compute optimal decisions for the prior 0.1

            minDCF_q = compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)
            actDCF_q = compute_actDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)

            minDCF_values_q.append(minDCF_q)
            actDCF_values_q.append(actDCF_q)

            print ('minDCF - pT = 0.5: %.4f' % minDCF_q)
            print ('actDCF - pT = 0.5: %.4f' % actDCF_q)
            print()

    plt.figure()
    plt.plot(np.logspace(-4, 2, 13),actDCF_values, label = 'actDCF')
    plt.plot(np.logspace(-4, 2, 13),minDCF_values, label = 'minDCF')
    plt.legend()
    plt.title('minDCF and actDCF in function of lambda')
    plt.xscale('log',base = 10)
    plt.savefig(f'LogReg/DCFplot_{mode}.png')

    plt.figure()
    plt.plot(np.logspace(-4, 2, 13),actDCF_values_w, label = 'actDCF')
    plt.plot(np.logspace(-4, 2, 13),minDCF_values_w, label = 'minDCF')
    plt.legend()
    plt.title('minDCF and actDCF in function of lambda')
    plt.xscale('log',base = 10)
    plt.savefig(f'LogReg/DCFplot_Weighted_{mode}.png')

    if mode != 'few_samples':
        plt.figure()
        plt.plot(np.logspace(-4, 2, 13),actDCF_values_q, label = 'actDCF')
        plt.plot(np.logspace(-4, 2, 13),minDCF_values_q, label = 'minDCF')
        plt.legend()
        plt.title('minDCF and actDCF in function of lambda')
        plt.xscale('log',base = 10)
        plt.savefig(f'LogReg/DCFplot_Quadratic_{mode}.png')
    
from confusion_matrix import (compute_optimal_Bayes_binary_llr, 
                                compute_confusion_matrix,
                                compute_empirical_Bayes_risk_binary,
                                compute_minDCF_binary_fast,
                                compute_Pfn_Pfp_allThresholds_fast)

def lab7(data,labels,model = "MVG", mode = "default", m = 0):
    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)
    llr = np.load("./backups/LLR_"+model+"_"+mode+".npy")

    return_value = 0.0

    if "pca" in mode:
        DTR,DVAL = apply_pca(DTR,DVAL,m=m)

    

    prior_cost_list = []

    
    prior_cost_list = [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0)]

    for prior, Cfn, Cfp in prior_cost_list:
        print()
        effective_prior = (prior*Cfn)/(prior*Cfn + (1-prior)*Cfp)
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp, '- Effective Prior ',effective_prior)
        
        predictions = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)

        print(compute_confusion_matrix(predictions, LVAL))
    
        print('DCF (non-normalized): %.3f' % (compute_empirical_Bayes_risk_binary(
            predictions, LVAL, prior, Cfn, Cfp, normalize=False)))
        DCF = compute_empirical_Bayes_risk_binary(predictions, LVAL, prior, Cfn, Cfp)
        print('DCF (normalized): %.3f' % (DCF))
        
        minDCF, minDCFThreshold = compute_minDCF_binary_fast(llr, LVAL, prior, Cfn, Cfp, returnThreshold=True)
        print('MinDCF (normalized, fast): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))
        return_value = (minDCF,DCF)

    # ROC plot - uncomment the commented lines to see the plot
    Pfn, Pfp, _ = compute_Pfn_Pfp_allThresholds_fast(llr, labels)
    plt.figure(0)
    plt.plot(Pfp, 1-Pfn)
    #plt.show()

    # Bayes error plot
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))


    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
        predictions = compute_optimal_Bayes_binary_llr(llr, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(predictions, LVAL, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(llr, LVAL, effPrior, 1.0, 1.0))
    plt.figure()
    plt.plot(effPriorLogOdds, actDCF, label='actDCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    plt.ylim([0, 1.1])
    plt.legend()

    title = f'Model = {model}'

    if m != 0:
        title += f' PCA m: {m}'

    plt.title(title)
    plt.savefig("DCF/DCF_" + model + "_" + mode + ".png")
    #plt.show()

    return return_value
            
def lab5(data, labels, mode = "default", m = 0):

    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)

    if "pca" in mode:
        
        DTR,DVAL = apply_pca(DTR,DVAL,m=m)

    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)

    for lab in [0,1]:
        #print('MVG - Class', lab)
        #print(hParams_MVG[lab][0])
        #print(hParams_MVG[lab][1])
        print()



    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])

    
    np.save("./backups/LLR_MVG_"+mode+".npy",LLR)

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

    np.save("./backups/LLR_Tied_"+mode+".npy",LLR)

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
    
    LLR = logpdf_GAU_ND(DVAL, hParams_Naive[1][0], hParams_Naive[1][1]) - logpdf_GAU_ND(DVAL, hParams_Naive[0][0], hParams_Naive[0][1])

    np.save("./backups/LLR_Naive_"+mode+".npy",LLR)

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

    show_hists(genuine,fake)
    
    #show_scatter(genuine,fake)

    show_hists(genuine_centered,fake_centered)

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
    show_hists_v2(y, labels,"./PCA/Hists")
    
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

        #y_pca,val_pca = apply_pca(DTR,DVAL,m)

        pca_matrix = compute_pca(DTR,m)
        y_pca = apply_pca_sol(pca_matrix,DTR)
        val_pca = apply_pca_sol(pca_matrix,DVAL)


        print()
        print("PCA TEST:\n",pca_matrix)
        print()

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

        print()

        U = compute_lda_JointDiag(y_pca,LTR,m=1)

        y_train = apply_lda(U,y_pca)


        #y_train = lda_matrix_train.T @ y_pca
        #y_val = lda_matrix_train.T @ val_pca


        print()
        print("LDA TEST:\n",U)
        print()

        if y_train[0, LTR==0].mean() > y_train[0, LTR==1].mean():
            U = -U
            y_train = U.T @ y_pca
        
        #y_val = lda_matrix_train.T @ val_pca
        y_val = apply_lda(U,val_pca)

        # Compute the treshold as the average of projected means
        threshold = (y_train[0, LTR==GENUINE].mean() + y_train[0, LTR==FAKE].mean()) / 2.0
    
        # Apply the threshold to the classification for the train dataset to test the accuracy
        
        accuracy = evaluate(y_train,LTR,threshold)
        print("m: ",m, " Error rate on train: ",100.0 - accuracy,"%", " Threshold: ", threshold )
        
        if accuracy > best_accuracy:
            print(accuracy,best_accuracy)
            best_accuracy = accuracy
            best_m = m
            best_threshold = threshold
            accuracy_val = evaluate(y_val,LVAL,threshold)
            
    print("Select m: ",best_m, " Error rate on val: ", 100.0 - accuracy_val, "%", " Threshold: ", best_threshold )
    
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