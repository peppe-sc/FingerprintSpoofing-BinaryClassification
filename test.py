
import numpy
import numpy as np





def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))


def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = vcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def parse_file(f):

    i = 0

    # For each line
    for line in f:
        
        # Split the line
        line = line.split(",")
        
        # Create a numpy array if first line
        if i == 0:
            dataset = np.array(line,dtype= np.float64)
            i = 1
        # Else stack new line
        else:
            dataset = np.vstack([dataset,np.array(line,dtype=np.float64)])
    
    # Transpose the dataset
    dataset = dataset.T

    # The labels are in the last row
    labels = dataset[-1].astype(np.int32)
    data = dataset[0:6]

    return data,labels


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)



def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):

    mu, C = compute_mu_C(D)
    U, s, Vh = np.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca_sol(P, D):
    return P.T @ D



f = open("trainData.txt","r")
#data,labels = parse_file(f)
#print(file)
f.close()

data,labels = load("trainData.txt")
#print(f)


(DTR, LTR), (DVAL, LVAL) = split_db_2to1(data, labels)


P = compute_pca(DTR,5)
print(P)
DTR_pca = apply_pca_sol(P,DTR)


