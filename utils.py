import numpy as np

def v_col(row):
    return row.reshape(row.size, 1)


def v_row(row):
    return row.reshape(1, row.size)

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

def center_data(data):

    # Calculate the mean and reshape as a column vector
    mean = data.mean(axis=1).reshape(data.shape[0],1)

    # Center data by subtracting the mean
    centered_data = data - mean
    
    return centered_data, mean

def compute_var_std(centered_data):
    # Calculate variance and standard deviation
    var = centered_data.var(axis=1).reshape(centered_data.shape[0],1)
    std = centered_data.std(axis=1).reshape(centered_data.shape[0],1)

    return var,std