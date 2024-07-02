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
from calibration import *


def lab11(DTR,LTR,DVAL,LVAL, pT = 0.1):

    best_actDCF = 9999999
    model = ""

    scores_LogReg = numpy.load('./backups/LogReg/LogReg_Quadratic.npy')
    scores_SVM = numpy.load('./backups/SVM/RBF.npy')
    scores_GMM = numpy.load('./backups/GMM/GMMscores.npy')

    scores_LogReg_eval = numpy.load('./backups/LogReg/eval_scores.npy')
    scores_SVM_eval = numpy.load('./backups/SVM/eval_scores.npy')
    scores_GMM_eval = numpy.load('./backups/GMM/eval_scores.npy')

    eval_data, eval_labels = parse_file(open("evalData.txt","r"))
    
    logreg_lam = numpy.load("./backups/LogReg/LogReg_lam.npy")

    C_SVM = numpy.load("./backups/SVM/C.npy")
    gamma_SVM = numpy.load("./backups/SVM/gamma.npy")
    
    numC_GMM = numpy.load("./backups/GMM/GMM_C.npy")
    cov_GMM = "diagonal"

    labels = LVAL

    fig = plt.figure(figsize=(16,9))
    axes = fig.subplots(3,3, sharex='all')
    fig.suptitle('K-fold')

    # We start with the computation of the system performance on the calibration set (whole dataset)
    print('LogReg: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        compute_minDCF_binary_fast(scores_LogReg, labels, 0.1, 1.0, 1.0),
        compute_actDCF_binary_fast(scores_LogReg, labels, 0.1, 1.0, 1.0)))

    print('SVM: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        compute_minDCF_binary_fast(scores_SVM, labels, 0.1, 1.0, 1.0),
        compute_actDCF_binary_fast(scores_SVM, labels, 0.1, 1.0, 1.0)))
    
    print('GMM: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        compute_minDCF_binary_fast(scores_GMM, labels, 0.1, 1.0, 1.0),
        compute_actDCF_binary_fast(scores_GMM, labels, 0.1, 1.0, 1.0)))

    # Comparison of actDCF / minDCF of both systems
    logOdds, actDCF, minDCF = bayesPlot(scores_LogReg, labels)
    axes[0,0].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
    axes[0,0].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF')

    logOdds, actDCF, minDCF = bayesPlot(scores_SVM, labels)
    axes[1,0].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF')
    axes[1,0].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'actDCF')

    logOdds, actDCF, minDCF = bayesPlot(scores_GMM, labels)
    axes[2,0].plot(logOdds, minDCF, color='C2', linestyle='--', label = 'minDCF')
    axes[2,0].plot(logOdds, actDCF, color='C2', linestyle='-', label = 'actDCF')
    
    axes[0,0].set_ylim(0, 0.8)    
    axes[0,0].legend()

    axes[1,0].set_ylim(0, 0.8)    
    axes[1,0].legend()

    axes[2,0].set_ylim(0, 0.8)    
    axes[2,0].legend()
    
    axes[0,0].set_title('LogReg - validation - non-calibrated scores')
    axes[1,0].set_title('SVM - validation - non-calibrated scores')
    axes[2,0].set_title('GMM - validation - non-calibrated scores')
    

    # Calibrating system indipendently

    # System 1
    calibrated_scores_LogReg = [] # We will add to the list the scores computed for each fold
    labels_LogReg = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.

    # We plot the non-calibrated minDCF and actDCF for reference
    logOdds, actDCF, minDCF = bayesPlot(scores_LogReg, labels)
    axes[0,1].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF (pre-cal.)')
    axes[0,1].plot(logOdds, actDCF, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    print ('LogReg')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.1), no cal.: %.3f' % compute_minDCF_binary_fast(scores_LogReg, labels, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % compute_actDCF_binary_fast(scores_LogReg, labels, 0.1, 1.0, 1.0))
    
    
    
    # Train KFOLD times the calibration model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_LogReg, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        calibrated_scores_LogReg.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labels_LogReg.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)
    calibrated_scores_LogReg = numpy.hstack(calibrated_scores_LogReg)
    labels_LogReg = numpy.hstack(labels_LogReg)

    act = compute_actDCF_binary_fast(calibrated_scores_LogReg, labels_LogReg, 0.1, 1.0, 1.0)

    # Evaluate the performance on pooled scores - we need to use the label vector labels_sys_1 since it's aligned to calibrated_scores_sys_1    
    print ('\t\tminDCF(p=0.1), cal.   : %.3f' % compute_minDCF_binary_fast(calibrated_scores_LogReg, labels_LogReg, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % act)
    
    if act < best_actDCF:
        best_actDCF = act 
        model = "LogReg"

    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_LogReg, labels_LogReg)
    axes[0,1].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF (cal.)') # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[0,1].legend()

    axes[0,1].set_title('LogReg - validation')
    axes[0,1].set_ylim(0, 0.8)    
    

    # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)
    # Eval
    w, b = trainWeightedLogRegBinary(vrow(scores_LogReg), labels, 0, pT)
    
    # We can use the trained model for application / evaluation data
    calibrated_eval_scores_LogReg = (w.T @ vrow(scores_LogReg_eval) + b - numpy.log(pT / (1-pT))).ravel()

    predictions = (calibrated_eval_scores_LogReg > 0).astype(int)  
    error_rate = np.mean(predictions != eval_labels) * 100.0

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.1)         : %.3f' % compute_minDCF_binary_fast(scores_LogReg_eval, eval_labels, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % compute_actDCF_binary_fast(scores_LogReg_eval, eval_labels, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % compute_actDCF_binary_fast(calibrated_eval_scores_LogReg, eval_labels, 0.1, 1.0, 1.0))    
    print ('\t\terror rate         : %.3f' % error_rate)
    
    # We plot minDCF, non-calibrated DCF and calibrated DCF for system 1
    logOdds, actDCF_precal, minDCF = bayesPlot(scores_LogReg_eval, eval_labels)
    logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores_LogReg, eval_labels) # minDCF is the same
    axes[0,2].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
    axes[0,2].plot(logOdds, actDCF_precal, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    axes[0,2].plot(logOdds, actDCF_cal, color='C0', linestyle='-', label = 'actDCF (cal.)')
    axes[0,2].set_ylim(0.0, 0.8)
    axes[0,2].set_title('LogReg - evaluation')
    axes[0,2].legend()


    # SVM
    calibrated_scores_SVM = [] # We will add to the list the scores computed for each fold
    labelsSVM = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.

    # We plot the non-calibrated minDCF and actDCF for reference
    logOdds, actDCF, minDCF = bayesPlot(scores_SVM, labels)
    axes[1,1].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF (pre-cal.)')
    axes[1,1].plot(logOdds, actDCF, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    print ('SVM')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.1), no cal.: %.3f' % compute_minDCF_binary_fast(scores_SVM, labels, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % compute_actDCF_binary_fast(scores_SVM, labels, 0.1, 1.0, 1.0))
    
    # We train the calibration model for the prior pT = 0.2
    
    # Train KFOLD times the calibration model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_SVM, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        calibrated_scores_SVM.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labelsSVM.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)
    calibrated_scores_SVM = numpy.hstack(calibrated_scores_SVM)
    labelsSVM = numpy.hstack(labelsSVM)

    act = compute_actDCF_binary_fast(calibrated_scores_SVM, labelsSVM, 0.1, 1.0, 1.0)

    # Evaluate the performance on pooled scores - we need to use the label vector labels_sys_1 since it's aligned to calibrated_scores_sys_1    
    print ('\t\tminDCF(p=0.1), cal.   : %.3f' % compute_minDCF_binary_fast(calibrated_scores_SVM, labelsSVM, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % act)

    if act < best_actDCF:
        best_actDCF = act
        model = "SVM"
    
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_SVM, labelsSVM)
    axes[1,1].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF (cal.)') # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[1,1].legend()

    axes[1,1].set_title('SVM - validation')
    axes[1,1].set_ylim(0, 0.8)

    # Eval
    w, b = trainWeightedLogRegBinary(vrow(scores_SVM), labels, 0, pT)
    
    # We can use the trained model for application / evaluation data
    calibrated_eval_scores_SVM = (w.T @ vrow(scores_SVM_eval) + b - numpy.log(pT / (1-pT))).ravel()

    predictions = (calibrated_eval_scores_SVM > 0).astype(int)  
    error_rate = np.mean(predictions != eval_labels) * 100.0

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.1)         : %.3f' % compute_minDCF_binary_fast(scores_SVM_eval, eval_labels, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % compute_actDCF_binary_fast(scores_SVM_eval, eval_labels, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % compute_actDCF_binary_fast(calibrated_eval_scores_SVM, eval_labels, 0.1, 1.0, 1.0))    
    print ('\t\terror rate         : %.3f' % error_rate)

    # We plot minDCF, non-calibrated DCF and calibrated DCF for system 1
    logOdds, actDCF_precal, minDCF = bayesPlot(scores_SVM_eval, eval_labels)
    logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores_SVM, eval_labels) # minDCF is the same
    axes[1,2].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
    axes[1,2].plot(logOdds, actDCF_precal, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    axes[1,2].plot(logOdds, actDCF_cal, color='C0', linestyle='-', label = 'actDCF (cal.)')
    axes[1,2].set_ylim(0.0, 0.8)
    axes[1,2].set_title('SVM - evaluation')
    axes[1,2].legend()


    # GMM
    calibrated_scores_GMM = [] # We will add to the list the scores computed for each fold
    labelsGMM = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.

    # We plot the non-calibrated minDCF and actDCF for reference
    logOdds, actDCF, minDCF = bayesPlot(scores_GMM, labels)
    axes[2,1].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF (pre-cal.)')
    axes[2,1].plot(logOdds, actDCF, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    print ('GMM')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.1), no cal.: %.3f' % compute_minDCF_binary_fast(scores_GMM, labels, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % compute_actDCF_binary_fast(scores_GMM, labels, 0.1, 1.0, 1.0))
    
    # We train the calibration model for the prior pT = 0.2
    
    # Train KFOLD times the calibration model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_GMM, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Train the model on the KFOLD - 1 training folds
         # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        calibrated_scores_GMM.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labelsGMM.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)
    calibrated_scores_GMM = numpy.hstack(calibrated_scores_GMM)
    labelsGMM = numpy.hstack(labelsGMM)

    act = compute_actDCF_binary_fast(calibrated_scores_GMM, labelsGMM, 0.1, 1.0, 1.0)

    # Evaluate the performance on pooled scores - we need to use the label vector labels_sys_1 since it's aligned to calibrated_scores_sys_1    
    print ('\t\tminDCF(p=0.1), cal.   : %.3f' % compute_minDCF_binary_fast(calibrated_scores_GMM, labelsGMM, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % act)

    if act < best_actDCF:
        best_actDCF = act
        model = "GMM"
    
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_GMM, labelsGMM)
    axes[2,1].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF (cal.)') # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[2,1].legend()

    axes[2,1].set_title('GMM - validation')
    axes[2,1].set_ylim(0, 0.8)


    # Eval
    w, b = trainWeightedLogRegBinary(vrow(scores_GMM), labels, 0, pT)
    
    # We can use the trained model for application / evaluation data
    calibrated_eval_scores_GMM = (w.T @ vrow(scores_GMM_eval) + b - numpy.log(pT / (1-pT))).ravel()

    predictions = (calibrated_eval_scores_GMM > 0).astype(int)  
    error_rate = np.mean(predictions != eval_labels) * 100.0

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.1)         : %.3f' % compute_minDCF_binary_fast(scores_GMM_eval, eval_labels, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % compute_actDCF_binary_fast(scores_GMM_eval, eval_labels, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % compute_actDCF_binary_fast(calibrated_eval_scores_GMM, eval_labels, 0.1, 1.0, 1.0))    
    print ('\t\terror rate         : %.3f' % error_rate)

    # We plot minDCF, non-calibrated DCF and calibrated DCF for system 1
    logOdds, actDCF_precal, minDCF = bayesPlot(scores_GMM_eval, eval_labels)
    logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores_GMM, eval_labels) # minDCF is the same
    axes[2,2].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
    axes[2,2].plot(logOdds, actDCF_precal, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    axes[2,2].plot(logOdds, actDCF_cal, color='C0', linestyle='-', label = 'actDCF (cal.)')
    axes[2,2].set_ylim(0.0, 0.8)
    axes[2,2].set_title('GMM - evaluation')
    axes[2,2].legend()

    #plt.show()
    tmp = str(pT).replace(".","_")
    plt.savefig(f"./CALIBRATION/comparison_{tmp}.png")


    # Fusion #
    

    fusedScores = [] # We will add to the list the scores computed for each fold
    fusedLabels = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.
    
    # We train the fusion for the prior 
    


    # Train KFOLD times the fusion model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training        
        SCAL1, SVAL1 = extract_train_val_folds_from_ary(scores_LogReg, foldIdx)
        SCAL2, SVAL2 = extract_train_val_folds_from_ary(scores_SVM, foldIdx)
        SCAL3, SVAL3 = extract_train_val_folds_from_ary(scores_GMM, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Build the training scores "feature" matrix
        SCAL = numpy.vstack([SCAL1, SCAL2, SCAL3])
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
        # Build the validation scores "feature" matrix
        SVAL = numpy.vstack([SVAL1, SVAL2, SVAL3])
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        fusedScores.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        fusedLabels.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)        
    fusedScores = numpy.hstack(fusedScores)
    fusedLabels = numpy.hstack(fusedLabels)

    # Evaluate the performance on pooled scores - we need to use the label vector fusedLabels since it's aligned to calScores_sys_2 (plot on same figure as system 1 and system 2)

    act = compute_actDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0)

    print ('Fusion')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.1)         : %.3f' % compute_minDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1)         : %.3f' % act)

    if act < best_actDCF:
        best_actDCF = act
        model = "Fusion"


    plt.figure()

    # As comparison, we select calibrated models trained with prior 0.2 (our target application)
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_LogReg, labels_LogReg)
    plt.title('Fusion - validation')
    plt.plot(logOdds, minDCF, color='C0', linestyle='--', label = 'S1 - minDCF')
    plt.plot(logOdds, actDCF, color='C0', linestyle='-', label = 'S1 - actDCF')
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_SVM, labelsSVM)
    plt.plot(logOdds, minDCF, color='C1', linestyle='--', label = 'S2 - minDCF')
    plt.plot(logOdds, actDCF, color='C1', linestyle='-', label = 'S2 - actDCF')
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_GMM, labelsGMM)
    plt.plot(logOdds, minDCF, color='C1', linestyle='--', label = 'S2 - minDCF')
    plt.plot(logOdds, actDCF, color='C1', linestyle='-', label = 'S2 - actDCF')      
    
    logOdds, actDCF, minDCF = bayesPlot(fusedScores, fusedLabels)
    plt.plot(logOdds, minDCF, color='C2', linestyle='--', label = 'Fusion - KFold - minDCF(0.2)')
    plt.plot(logOdds, actDCF, color='C2', linestyle='-', label = 'Fusion - KFold - actDCF(0.2)')
    #plt.set_ylim(0.0, 0.8)
    plt.legend()

    #plt.show()
    tmp = str(pT).replace(".","_")
    plt.savefig(f"./CALIBRATION/fusion_val_{tmp}.png")

    # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)

    SMatrix = numpy.vstack([scores_LogReg, scores_SVM, scores_GMM])
    w, b = trainWeightedLogRegBinary(SMatrix, labels, 0, pT)

    # Apply model to application / evaluation data
    SMatrixEval = numpy.vstack([scores_LogReg_eval, scores_SVM_eval, scores_GMM_eval])
    fused_eval_scores = (w.T @ SMatrixEval + b - numpy.log(pT / (1-pT))).ravel()

    predictions = (fused_eval_scores > 0).astype(int)  
    error_rate = np.mean(predictions != eval_labels) * 100.0

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.1)         : %.3f' % compute_minDCF_binary_fast(fused_eval_scores, eval_labels, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1)         : %.3f' % compute_actDCF_binary_fast(fused_eval_scores, eval_labels, 0.1, 1.0, 1.0))
    print ('\t\terror rate         : %.3f' % error_rate)
    
    plt.figure()

    # We plot minDCF, actDCF for calibrated system 1, calibrated system 2 and fusion
    #logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_scores_LogReg, eval_labels)
    #plt.plot(logOdds, minDCF, color='C0', linestyle='--', label = 'S1 - minDCF')
    #plt.plot(logOdds, actDCF, color='C0', linestyle='-', label = 'S1 - actDCF')
    #logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_scores_SVM, eval_labels)
    #plt.plot(logOdds, minDCF, color='C1', linestyle='--', label = 'S2 - minDCF')
    #plt.plot(logOdds, actDCF, color='C1', linestyle='-', label = 'S2 - actDCF')
    #logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_scores_GMM, eval_labels)
    #plt.plot(logOdds, minDCF, color='C1', linestyle='--', label = 'S2 - minDCF')
    #plt.plot(logOdds, actDCF, color='C1', linestyle='-', label = 'S2 - actDCF')
    
    logOdds, actDCF, minDCF = bayesPlot(fused_eval_scores, eval_labels) # minDCF is the same
    plt.plot(logOdds, minDCF, color='C2', linestyle='--', label = 'Fusion - minDCF')
    plt.plot(logOdds, actDCF, color='C2', linestyle='-', label = 'Fusion - actDCF')
    #plt.set_ylim(0.0, 0.8)
    plt.title('Fusion - evaluation')
    plt.legend()


    #plt.show()
    tmp = str(pT).replace(".","_")
    plt.savefig(f"./CALIBRATION/fusion_eval_{tmp}.png")

    return best_actDCF, model


def main():
    # Open the file
    f = open("trainData.txt","r")

    # Parse the file
    data,labels = parse_file(f)
    f.close()
    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)

    best_pT = 0.1
    best_actDCF = 99999999
    best_model = ""
    for pT in [0.1,0.4,0.5,0.9]:

        actDCF, model = lab11(DTR,LTR,DVAL,LVAL, pT)

        if actDCF < best_actDCF:
            best_actDCF = actDCF
            best_model = model
            best_pT = pT

    print(f'The best performances on validation set are achieved by the {best_model} model, with actDCF = {best_actDCF} and pT = {best_pT}')

if __name__ == "__main__":
    
    main()