from utils import *
import matplotlib.pyplot as plt


GENUINE = 1
FAKE = 0

def find_best_threshold(train_data,initial_threshold,LTR):
    threshold_list = np.linspace(initial_threshold-(initial_threshold/2.0),initial_threshold+(initial_threshold/2.0),num=300)
    best_t = initial_threshold
    PVAL = np.zeros(shape=LTR.shape, dtype=np.int32)
    PVAL[train_data[0] >= initial_threshold] = GENUINE
    PVAL[train_data[0] < initial_threshold] = FAKE

    errors = PVAL == LTR
    errors_count = 0

    for e in errors:
        if not e:
            errors_count += 1
    best_accuracy = 100.0*(errors.shape[0]-errors_count)/float(errors.shape[0])
    
    for threshold in threshold_list:
        
        PVAL = np.zeros(shape=LTR.shape, dtype=np.int32)
        PVAL[train_data[0] >= threshold] = GENUINE
        PVAL[train_data[0] < threshold] = FAKE

        errors = PVAL == LTR
        errors_count = 0

        for e in errors:
             if not e:
                  errors_count += 1
        
        if 100.0*(errors.shape[0]-errors_count)/float(errors.shape[0]) > best_accuracy:
            best_accuracy = 100.0*(errors.shape[0]-errors_count)/float(errors.shape[0])
            best_t = threshold
        #print("With a threshold of: ",threshold, " Accuracy: ",100.0*(errors.shape[0]-errors_count)/float(errors.shape[0]),"%")
    return best_t, 100.0 - best_accuracy

def show_hists_v2(data, labels, folder):
    # For each feature
    for feature in range(6):
        # Create the histogram for the selected feature
        plt.figure()
        plt.hist(data[:,labels == GENUINE][feature], density=True, alpha = 0.5, label= "Genuine",bins= 30)
        plt.hist(data[:,labels == FAKE][feature], density=True, alpha = 0.5, label= "Fake",bins= 30)

        plt.legend(loc = "upper right")
        plt.xlabel("Feature " + str(feature))
        plt.title("Feature %d" % (feature + 1))
        plt.savefig(folder + "/hist" + str(feature) + ".png")
    #plt.show()


from lda import *
from pca import *

def lab3(data, labels):

    genuine = data[:,labels == GENUINE]
    fake = data[:,labels == FAKE]

    # Compute the mean for the genuine class and center the data
    genuine_centered, mean_genuine = center_data(genuine)

    # Compute the mean for the fake class and center the data
    fake_centered, mean_fake = center_data(fake)

    #Lab 3

    # Center data
    centered_data, mean = center_data(data)

    # Compute the covariance matrix
    covariance_matrix = (centered_data @ centered_data.T)/float(data.shape[1])
    print("Covariance matrix:\n",covariance_matrix)

    # Compute eigenvalue and eigenvectors
    eigenvalues,eigenvectors = np.linalg.eigh(covariance_matrix)
    print("Eigenvalues:\n",eigenvalues) # They are all > 0
    # Sort the eigenvectors in descending order
    P = eigenvectors[:,::-1]

    # Compute the projection of the entire dataset in the new space
    y = np.dot(P.T,data)

    # Plot the histograms
    show_hists_v2(y, labels,"./PCA/Hists")
    
    # Compute the within class covariance matrix 
    Sw = compute_SW(genuine,genuine_centered,fake,fake_centered,data)
    print("Sw:\n", Sw)

    # Compute the between class covariance matrix 
    Sb = compute_SB(genuine,mean_genuine,fake,mean_fake,mean,data)
    print("Sb:\n", Sb)

    # Compute Singular Value Decomposition for Sw
    U,s,_ = np.linalg.svd(Sw)

    # P1 can be calculated as follows
    P1 = np.dot(U*v_row(1.0/(s**0.5)),U.T)

    # Compute the transformed between class covariance
    Sbt = P1 @ Sb @ P1.T
    print("Sbt:\n",Sbt)
    
    # Obtain the eigenvalues and eigenvectors of Sbt
    lda_eigenvalues,lda_eigenvectors = np.linalg.eigh(Sbt)

    # we have only one class so apply lda with 1 dimension
    m = 1

    # Sort the eigenvectors and take the m highest
    P2 = lda_eigenvectors[:,::-1][:, 0:m]
    print(lda_eigenvectors)

    # W is the lda matrix
    W = P1.T @ P2
    print("W:\n",W)

    # The single dimension data for lda
    y_lda = W.T @ data

    # Histogram of lda, the classes overlap a bit
    plt.figure()
    plt.hist(y_lda[:,labels == GENUINE][0], density=True, alpha = 0.5, label= "Genuine",bins= 30)
    plt.hist(y_lda[:,labels == FAKE][0], density=True, alpha = 0.5, label= "Fake",bins= 30)

    plt.legend(loc = "upper right")
    #plt.xlabel("Feature " + str(0))
    plt.title("LDA Histogram")
    plt.savefig("LDA/Hists/hist" + str(0) + ".png")

    #plt.show()

    # Split in 2/3 for training and 1/3 for val
    (DTR,LTR),(DVAL,LVAL) = split_db_2to1(data,labels)

    # Split the train dataset and compute the train mean and the mean per class
    genuine_train = DTR[:,LTR == GENUINE]
    fake_train = DTR[:,LTR == FAKE]

    genuine_centered_train, mean_genuine_train = center_data(genuine_train)
    fake_centered_train, mean_fake_train = center_data(fake_train)
    DTR_centered, mean_train = center_data(DTR)

    # Compute the within class covariance
    Sw_train = compute_SW(genuine_train,genuine_centered_train,fake_train,fake_centered_train,DTR)
    print("Sw train:\n",Sw_train)
    
    # Compute the between class covariance matrix
    Sb_train = compute_SB(genuine_train,mean_genuine_train,fake_train,mean_fake_train,mean_train,DTR)
    print("Sb train:\n",Sb_train)

    # Compute lda matrix and ptoject the training and val data
    lda_matrix_train = compute_lda_matrix(Sw_train,Sb_train)
    print("Lda matrix train:\n",lda_matrix_train)

    y_train = lda_matrix_train.T @ DTR
    y_val = lda_matrix_train.T @ DVAL

    
    plt.figure()
    plt.hist(y_train[:,LTR == GENUINE][0], density=True, alpha = 0.5, label= "Genuine",bins= 30)
    plt.hist(y_train[:,LTR == FAKE][0], density=True, alpha = 0.5, label= "Fake",bins= 30)

    plt.legend(loc = "upper right")
    plt.title("Train")

    plt.figure()
    plt.hist(y_val[:,LVAL == GENUINE][0], density=True, alpha = 0.5, label= "Genuine",bins= 30)
    plt.hist(y_val[:,LVAL == FAKE][0], density=True, alpha = 0.5, label= "Fake",bins= 30)

    plt.legend(loc = "upper right")
    plt.title("Val")
    #plt.show()
    
    # Compute the treshold as the average of projected means
    threshold = (y_train[0, LTR==GENUINE].mean() + y_train[0, LTR==FAKE].mean()) / 2.0
    
    # Apply the threshold to the classification for the val dataset to test the accuracy
    accuracy = evaluate(y_val,LVAL,threshold)
   
    print("With an initial threshold of: ",threshold, " Error rate on val: ",100.0 - accuracy,"%")
    
    
    
    # Try several values for the threshold in the training data
    threshold, error_rate = find_best_threshold(y_train,threshold,LTR)
    print("With a threshold of: ",threshold, " Error rate on train: ",error_rate,"%")

    # Evaluate the final threshold usign validation set
    accuracy = evaluate(y_val,LVAL,threshold)
    
    print("With a threshold of: ",threshold, " Error rate on val: ",100.0 - accuracy,"%")
    best_accuracy = 0.0
    best_m = 0
    best_threshold = 0.0
    
    for m in range(2,6):

        #y_pca,val_pca = apply_pca(DTR,DVAL,m)

        pca_matrix = compute_pca(DTR,m)
        y_pca = apply_pca_sol(pca_matrix,DTR)
        val_pca = apply_pca_sol(pca_matrix,DVAL)


        #print()
        #print("PCA TEST:\n",pca_matrix)
        #print()

        # Split the train dataset and compute the train mean and the mean per class
        genuine_train = y_pca[:,LTR == GENUINE]
        fake_train = y_pca[:,LTR == FAKE]

        genuine_centered_train, mean_genuine_train = center_data(genuine_train)
        fake_centered_train, mean_fake_train = center_data(fake_train)
        DTR_centered, mean_train = center_data(y_pca)

        # Compute the within class covariance
        Sw_train = compute_SW(genuine_train,genuine_centered_train,fake_train,fake_centered_train,y_pca)

        # Compute the between class covariance matrix
        Sb_train = compute_SB(genuine_train,mean_genuine_train,fake_train,mean_fake_train,mean_train,y_pca)

        # Compute lda matrix and ptoject the training and val data
        lda_matrix_train = compute_lda_matrix(Sw_train,Sb_train)

        print()

        U = compute_lda_JointDiag(y_pca,LTR,m=1)

        y_train = apply_lda(U,y_pca)


        #y_train = lda_matrix_train.T @ y_pca
        #y_val = lda_matrix_train.T @ val_pca


        #print()
        #print("LDA TEST:\n",U)
        #print()

        if y_train[0, LTR==0].mean() > y_train[0, LTR==1].mean():
            U = -U
            y_train = U.T @ y_pca
        
        #y_val = lda_matrix_train.T @ val_pca
        y_val = apply_lda(U,val_pca)

        # Compute the treshold as the average of projected means
        threshold = (y_train[0, LTR==GENUINE].mean() + y_train[0, LTR==FAKE].mean()) / 2.0
    
        # Apply the threshold to the classification for the train dataset to test the accuracy
        
        accuracy = evaluate(y_train,LTR,threshold)
        print("m: ",m, " Error rate on train: ",100.0 - accuracy,"%", " Threshold: ", threshold )
        
        if accuracy > best_accuracy:
            print(accuracy,best_accuracy)
            best_accuracy = accuracy
            best_m = m
            best_threshold = threshold
            accuracy_val = evaluate(y_val,LVAL,threshold)
            
    print("Select m: ",best_m, " Error rate on val: ", 100.0 - accuracy_val, "%", " Threshold: ", best_threshold )
    

    return


def main():
    # Open the file
    f = open("trainData.txt","r")

    # Parse the file
    data,labels = parse_file(f)

    lab3(data,labels)




if __name__ == "__main__":
    
    main()