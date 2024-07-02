from utils import *
import matplotlib.pyplot as plt


GENUINE = 1
FAKE = 0


from gau import *
from pca import *
from lda import *
from confusion_matrix import *

def lab7(DTR,LTR,DVAL,LVAL,model = "MVG", mode = "default", m = 0):
    
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
    Pfn, Pfp, _ = compute_Pfn_Pfp_allThresholds_fast(llr, LVAL)
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
    

def main():
    # Open the file
    f = open("trainData.txt","r")

    # Parse the file
    data,labels = parse_file(f)

    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)
    
    prior_cost_list = [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]

    for prior, Cfn, Cfp in prior_cost_list:
        effective_prior = (prior*Cfn)/(prior*Cfn + (1-prior)*Cfp)
        print(f'Effective prior for configuration {prior} {Cfn} {Cfp} = ', effective_prior)

    # Lab7

    best_m_MVG = 0
    best_DCF_MVG = 99999999999999.0

    
    best_m_Tied = 0
    best_DCF_Tied = 99999999999999.0

    
    best_m_Naive = 0
    best_DCF_Naive = 99999999999999.0
    

    print("\n>>> START: Lab 7 MVG\n")
    lab7(DTR,LTR,DVAL,LVAL)
    print("\n>>> END: Lab 7 MVG\n")
    
    print("\n>>> START: Lab 7 Tied\n")
    lab7(DTR,LTR,DVAL,LVAL, model="Tied")
    print("\n>>> END: Lab 7 Tied\n")

    print("\n>>> START: Lab 7 Naive\n")
    lab7(DTR,LTR,DVAL,LVAL, model="Naive")
    print("\n>>> END: Lab 7 Naive\n")

    for m in range(1,6):

        print("\n>>> START: Lab 7 PCA MVG\n", "m = ", m)

        dcf,actDCF = lab7(DTR,LTR,DVAL,LVAL, mode="pca_m"+str(m),m=m)

        if(dcf < best_DCF_MVG):
            best_m_MVG = m
            best_DCF_MVG = dcf
            best_act_MVG = actDCF

        print("\n>>> END: Lab 7 PCA MVG\n")

        print("\n>>> START: Lab 7 PCA Tied\n" "m = ", m)
        dcf,actDCF = lab7(DTR,LTR,DVAL,LVAL, model="Tied", mode="pca_m"+str(m), m=m)
        if(dcf < best_DCF_Tied):
            best_m_Tied = m
            best_DCF_Tied = dcf
            best_act_Tied = actDCF
        print("\n>>> END: Lab 7 PCA Tied\n")

        print("\n>>> START: Lab 7 PCA Naive\n" "m = ", m)
        dcf,actDCF = lab7(DTR,LTR,DVAL,LVAL, model="Naive", mode="pca_m"+str(m), m=m)
        if(dcf < best_DCF_Naive):
            best_m_Naive = m
            best_DCF_Naive = dcf
            best_act_Naive = actDCF
        print("\n>>> END: Lab 7 PCA Naive\n")

        #print("The best model for prior 0.1 with PCA m = ", best_m, ", model = ", best_model, "minDCF = ", best_DCF)

    print(f'MVG with PCA, best m = {best_m_MVG}, minDCF = {best_DCF_MVG}, actDCF = {best_act_MVG}, Calibration error: {round(100.0*(best_act_MVG-best_DCF_MVG)/best_DCF_MVG,2)}%')
    print(f'Tied with PCA, best m = {best_m_Tied}, minDCF = {best_DCF_Tied}, actDCF = {best_act_Tied}, Calibration error: {round(100.0*(best_act_Tied-best_DCF_Tied)/best_DCF_Tied,2)}%')
    print(f'Naive with PCA, best m = {best_m_Naive}, minDCF = {best_DCF_Naive}, actDCF = {best_act_Naive}, Calibration error: {round(100.0*(best_act_Naive-best_DCF_Naive)/best_DCF_Naive,2)}%')



if __name__ == "__main__":
    
    main()