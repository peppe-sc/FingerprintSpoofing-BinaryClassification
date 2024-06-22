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

best_minDCF = 9999999999.0
best_llr = None
model = ""
best_C = 0.0
best_gamma = 0.0
eval_llr = None


def lab9(DTR,LTR,DVAL,LVAL):


    global best_minDCF
    global best_llr
    global model 
    global best_C
    global best_gamma
    global eval_llr

    C_values = np.logspace(-5, 0, 11)
    
    minDCF_values = []
    actDCF_values = []

    K = 1.0

    eval_data, _ = parse_file(open("./evalData.txt","r"))

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

        if minDCF < best_minDCF:
            best_minDCF = minDCF
            best_llr = SVAL
            model = "Linear"
            best_C = C
            eval_llr = (v_row(w) @ eval_data + b).ravel()

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

        if minDCF < best_minDCF:
            best_minDCF = minDCF
            best_llr = SVAL
            model = "Poly"
            best_C = C
            eval_llr = fScore(eval_data)


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

            if minDCF < best_minDCF:
                best_minDCF = minDCF
                best_llr = SVAL
                model = "RBF"
                best_C = C
                best_gamma = gamma
                eval_llr = fScore(eval_data)

        plt.plot(C_values,minDCF_values, label = f'minDCF g: {gamma:.2f}')
        plt.plot(C_values,actDCF_values, label = f'actDCF g: {gamma:.2f}')

    plt.legend()
    plt.title('minDCF and actDCF grid search C,gamma')
    plt.xscale('log',base = 10)
    plt.savefig("SVM/rbf_svm.png")


def main():
    # Open the file
    f = open("trainData.txt","r")

    # Parse the file
    data,labels = parse_file(f)

    f.close()

    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)


    lab9(DTR,LTR,DVAL,LVAL)

    np.save(f"./backups/SVM/{model}.npy",best_llr)
    np.save("./backups/SVM/C.npy", best_C)
    np.save("./backups/SVM/gamma.npy", best_gamma)
    np.save("./backups/SVM/eval_scores.npy",eval_llr)
    print(f"Best model for SVM {model} with minDCF = {best_minDCF}")


if __name__ == "__main__":
    
    main()