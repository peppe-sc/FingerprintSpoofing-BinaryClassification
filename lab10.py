from utils import *
import matplotlib.pyplot as plt

GENUINE = 1
FAKE = 0


from gau import *
from pca import *
from lda import *
from confusion_matrix import *
from logreg import *
from svm import *
from gmm import *

def lab10(DTR,LTR,DVAL,LVAL):

    best_minDCF = 9999999999.0
    best_llr = None
    model = ""
    best_cov = ""
    best_numC = 1
    eval_score = None

    eval_data, _ = parse_file(open("evalData.txt","r"))

    for covType in ['full', 'diagonal', 'tied']:
        print(covType)
        for numC in [1,2,4,8,16,32]:
            gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, covType = covType, verbose=False, psiEig = 0.01)
            gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC, covType = covType, verbose=False, psiEig = 0.01)

            SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
            print(numC, covType)
            #print ('minDCF - pT = 0.5: %.4f' % bayesRisk.compute_minDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0))
            #print ('actDCF - pT = 0.5: %.4f' % bayesRisk.compute_actDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0))
            minDCF = compute_minDCF_binary_fast(SLLR, LVAL, 0.1, 1.0, 1.0)
            print ('\tnumC = %d: %.4f / %.4f' % (numC, minDCF, compute_actDCF_binary_fast(SLLR, LVAL, 0.1, 1.0, 1.0)))
            
            predictions = (SLLR > 0).astype(int)  
            error_rate = np.mean(predictions != LVAL) * 100.0
            print(f'\tError rate: {error_rate:.4f}%')

            if minDCF < best_minDCF:
                best_minDCF = minDCF
                best_llr = SLLR
                model = f"GMM{numC}"
                best_cov = covType
                best_numC = numC

                eval_score = logpdf_GMM(eval_data, gmm1) - logpdf_GMM(eval_data, gmm0)


        print()

    np.save("./backups/GMM/GMMscores.npy",best_llr)
    np.save("./backups/GMM/GMMcov.npy",best_cov)
    np.save("./backups/GMM/GMM_C.npy",best_numC)
    np.save("./backups/GMM/eval_scores.npy",eval_score)

    effPriorLogOdds = np.linspace(-4, 4, 30)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    # Bayes error plot GMM
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
        predictions = compute_optimal_Bayes_binary_llr(best_llr, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(predictions, LVAL, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(best_llr, LVAL, effPrior, 1.0, 1.0))
    plt.figure()
    plt.plot(effPriorLogOdds, actDCF, label='actDCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    plt.ylim([0, 1.1])
    plt.legend()
    plt.title("GMM")
    plt.savefig(f"./GMM/bayesplot{model}.png")


    # Bayes error plot LogReg

    best_llr = np.load("./backups/LogReg/LogReg_Quadratic.npy")

    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
        predictions = compute_optimal_Bayes_binary_llr(best_llr, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(predictions, LVAL, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(best_llr, LVAL, effPrior, 1.0, 1.0))
    plt.figure()
    plt.plot(effPriorLogOdds, actDCF, label='actDCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    plt.ylim([0, 1.1])
    plt.legend()
    plt.title("LogReg")
    plt.savefig(f"./GMM/bayesplotLogReg.png")

    
    # Bayes error plot SVM

    best_llr = np.load("./backups/SVM/RBF.npy")

    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
        predictions = compute_optimal_Bayes_binary_llr(best_llr, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(predictions, LVAL, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(best_llr, LVAL, effPrior, 1.0, 1.0))
    plt.figure()
    plt.plot(effPriorLogOdds, actDCF, label='actDCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    plt.ylim([0, 1.1])
    plt.legend()
    plt.title("SVM")
    plt.savefig(f"./GMM/bayesplotSVM.png")







def main():
    # Open the file
    f = open("trainData.txt","r")

    # Parse the file
    data,labels = parse_file(f)

    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)

    lab10(DTR,LTR,DVAL,LVAL)


if __name__ == "__main__":
    
    main()