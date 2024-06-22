import numpy as np
from utils import *
import scipy

GENUINE = 1
FAKE = 0

def compute_SB(genuine,mean_genuine,fake,mean_fake,mean,data):
    Sb = (float(genuine.shape[1])*(mean_genuine-mean) @ ((mean_genuine-mean).T) + float(fake.shape[1])*(mean_fake-mean) @ ((mean_fake-mean).T))/data.shape[1]
    return Sb

def compute_SW(genuine,genuine_centered,fake,fake_centered,data):
    # Compute the covariance per class
    Sw_genuine = (genuine_centered @ genuine_centered.T)/float(genuine.shape[1])
    Sw_fake = (fake_centered @ fake_centered.T)/float(fake.shape[1])

    # Compute the within class covariance matrix 
    Sw = (float(genuine.shape[1])*Sw_genuine + float(fake.shape[1])*Sw_fake)/data.shape[1]
    return Sw

def compute_lda_matrix(Sw,Sb,m=1):
    # Compute Singular Value Decomposition for Sw
    U,s,_ = np.linalg.svd(Sw)

    # P1 can be calculated as follows
    P1 = np.dot(U*v_row(1.0/(s**0.5)),U.T)

    # Compute the transformed between class covariance
    Sbt = P1 @ Sb @ P1.T
    #print("Sbt:\n",Sbt)
    
    # Obtain the eigenvalues and eigenvectors of Sbt
    lda_eigenvalues,lda_eigenvectors = np.linalg.eigh(Sbt)

    # we have only two classes so apply lda with 1 dimension
    

    # Sort the eigenvectors and take the m highest
    P2 = lda_eigenvectors[:,::-1][:, 0:m]
    #print(lda_eigenvectors)

    # W is the lda matrix
    W = P1.T @ P2

    return W


def evaluate(data,labels,threshold):
    
    # Apply the threshold to the classification for the val dataset to test the accuracy
    PVAL = np.zeros(shape=labels.shape, dtype=np.int32)
    PVAL[data[0] >= threshold] = GENUINE
    PVAL[data[0] < threshold] = FAKE

    errors = PVAL == labels
    errors_count = 0

    for e in errors:
        if not e:
            errors_count += 1
    print(f'Errors: {errors_count}\tTotal: {data.shape[1]}')
    # Return the accuracy
    return 100.0*(errors.shape[0]-errors_count)/float(errors.shape[0])

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = v_col(D.mean(1))
    for i in np.unique(L):
        DCls = D[:, L == i]
        mu = v_col(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L, m):
    
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def compute_lda_JointDiag(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)

    U, s, _ = np.linalg.svd(Sw)
    P = np.dot(U * v_row(1.0/(s**0.5)), U.T)

    Sb2 = np.dot(P, np.dot(Sb, P.T))
    U2, s2, _ = np.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return np.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D
