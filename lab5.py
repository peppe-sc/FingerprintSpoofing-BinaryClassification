from utils import *
import matplotlib.pyplot as plt


GENUINE = 1
FAKE = 0


from gau import *
from pca import *
from lda import *

def lab5(DTR, LTR, DVAL, LVAL, mode = "default", m = 0):

    

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

    if mode == "default":


        C = hParams_MVG[0][1]
        Corr_0 = C / ( v_col(C.diagonal()**0.5) * v_row(C.diagonal()**0.5) )
        print("Correlation Matrix for class 0:\n",Corr_0)
    
        C = hParams_MVG[1][1]
        Corr_1 = C / ( v_col(C.diagonal()**0.5) * v_row(C.diagonal()**0.5) )
        print("Correlation Matrix for class 1:\n",Corr_1)


def main():
    # Open the file
    f = open("trainData.txt","r")

    # Parse the file
    data,labels = parse_file(f)

    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)

    print("\n>>> START: Lab 5 with all the feature\n")
    lab5(DTR,LTR,DVAL,LVAL)
    print("\n>>> END: Lab 5 with all the feature\n")

    print("\n>>> START: Lab 5 with only the first 4 features\n")
    lab5(DTR[0:4,:],LTR,DVAL[0:4,:],LVAL, mode="first_4")
    print("\n>>> END: Lab 5 with only the first 4 features\n")

    print("\n>>> START: Lab 5 with only the first 2 features\n")
    lab5(DTR[0:2,:],LTR,DVAL[0:2,:],LVAL, mode="first_2")
    print("\n>>> END: Lab 5 with only the first 2 features\n")

    print("\n>>> START: Lab 5 with only the features 3-4\n")
    lab5(DTR[2:4,:],LTR,DVAL[2:4,:],LVAL, mode="last_2")
    print("\n>>> END: Lab 5 with only features 3-4\n")
    
    for m in range(1,6):
        print("\n>>> START: Lab 5 with PCA m = "+ str(m) +"\n")
        lab5(DTR,LTR,DVAL,LVAL, mode="pca_m"+str(m), m = m)
        print("\n>>> END: Lab 5 with PCA m = " + str(m) + "\n")





if __name__ == "__main__":
    
    main()