import numpy
from utils import *
import scipy
# Optimize the logistic regression loss
def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(v_col(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (v_row(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

# Optimize the weighted logistic regression loss
def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTrue = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wFalse = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(v_col(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTrue # Apply the weights to the loss computations
        loss[ZTR<0] *= wFalse

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTrue # Apply the weights to the gradient computations
        G[ZTR < 0] *= wFalse
        
        GW = (v_row(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]



def quadratic_features(D):
    n_features, n_samples = D.shape
    quadratic_terms = []
    for i in range(n_features):
        for j in range(i, n_features):
            quadratic_terms.append(D[i] * D[j])
    return numpy.vstack([D] + quadratic_terms)

def trainLogRegQuadratic(DTR, LTR, l):
    ZTR = LTR * 2.0 - 1.0  # Convert labels to -1 and 1

    # Generate quadratic features
    DTR_quad = quadratic_features(DTR)

    def logreg_obj_with_grad(v):  # Compute both the objective and its gradient
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(v_col(w).T, DTR_quad).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (v_row(G) * DTR_quad).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    initial_v = numpy.zeros(DTR_quad.shape[0] + 1)
    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0=initial_v)[0]
    print("Log-reg Quadratic - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]