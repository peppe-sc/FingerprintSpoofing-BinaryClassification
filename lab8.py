from utils import *
import matplotlib.pyplot as plt

GENUINE = 1
FAKE = 0


from gau import *
from pca import *
from lda import *
from confusion_matrix import *
from logreg import *

best_minDCF = 9999999999.0
best_llr = None
model = ""
best_lam = 0.0
eval_scores = None

def lab8(DTR,LTR,DVAL,LVAL, mode = 'default'):

    global best_minDCF
    global best_llr  
    global model 
    global best_lam
    global eval_scores

    eval_data, _ = parse_file(open("./evalData.txt","r"))

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

        print ('minDCF - pT = 0.1: %.4f' % minDCF)
        print ('actDCF - pT = 0.1: %.4f' % actDCF)
        print()

        if minDCF < best_minDCF:
            best_minDCF = minDCF
            best_llr = sValLLR
            model = "Linear"
            best_lam = lam

            sVal = np.dot(w.T, eval_data) + b
            sValLLR = sVal - np.log(pEmp / (1-pEmp))
            eval_scores = sValLLR


        if mode == "center":
            continue

        # Weighted
        pT = 0.1
        w, b = trainWeightedLogRegBinary(DTR, LTR, lam, pT = pT) # Train model to print the loss
        sVal = np.dot(w.T, DVAL) + b
        sValLLR = sVal - np.log(pT / (1-pT))

        minDCF_w = compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
        actDCF_w = compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)

        minDCF_values_w.append(minDCF_w)
        actDCF_values_w.append(actDCF_w)

        print ('minDCF - pT = 0.1: %.4f' % minDCF_w)
        print ('actDCF - pT = 0.1: %.4f' % actDCF_w)
        
        if minDCF_w < best_minDCF:
            best_minDCF = minDCF_w
            best_llr = sValLLR
            model = "Weighted"
            best_lam = lam

            sVal = np.dot(w.T, eval_data) + b
            sValLLR = sVal - np.log(pT / (1-pT))
            eval_scores = sValLLR

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

            print ('minDCF - pT = 0.1: %.4f' % minDCF_q)
            print ('actDCF - pT = 0.1: %.4f' % actDCF_q)
            print()

            if minDCF_q < best_minDCF:
                best_minDCF = minDCF_q
                best_llr = sValLLR
                model = "Quadratic"
                best_lam = lam

                EVAL_quad = quadratic_features(eval_data)
                sVal = np.dot(w.T, EVAL_quad) + b
                sValLLR = sVal - np.log(pEmp / (1-pEmp))
                eval_scores = sValLLR




    plt.figure()
    plt.plot(np.logspace(-4, 2, 13),actDCF_values, label = 'actDCF')
    plt.plot(np.logspace(-4, 2, 13),minDCF_values, label = 'minDCF')
    plt.legend()
    plt.title('minDCF and actDCF in function of lambda')
    plt.xscale('log',base = 10)
    plt.savefig(f'LogReg/DCFplot_{mode}.png')

    if mode == "center":
        return

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
    


def main():
    # Open the file
    f = open("trainData.txt","r")

    # Parse the file
    data,labels = parse_file(f)
    f.close()
    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)

    print("\n>>> START: Lab 8 \n")
    lab8(DTR,LTR,DVAL,LVAL)
    print("\n>>> END: Lab 8 \n")

    print("\n>>> START: Lab 8 only few samples\n")
    lab8(DTR[:,::50],LTR[::50],DVAL,LVAL, mode='few_samples')
    print("\n>>> END: Lab 8 only few samples\n")

    DTR_centered, mean = center_data(DTR)


    print("\n>>> START: Lab 8 centered dataset\n")
    lab8(DTR_centered,LTR,DVAL - mean,LVAL, mode="center")
    print("\n>>> END: Lab 8 centered dataset\n")

    np.save(f"./backups/LogReg/LogReg_{model}.npy",best_llr)
    np.save("./backups/LogReg/LogReg_lam.npy", best_lam)
    np.save("./backups/LogReg/eval_scores.npy", eval_scores)
    print(f"Best minDCF obtained by {model} with minDCF = {best_minDCF}")

if __name__ == "__main__":
    
    main()





"""BEST
Log-reg Quadratic - lambda = 3.162278e-02 - J*(w, b) = 2.687671e-01
Quadratic shape val:  (27, 2000)
Error rate: 5.9
minDCF - pT = 0.5: 0.2436
actDCF - pT = 0.5: 0.4952
"""