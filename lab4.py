from utils import *
import matplotlib.pyplot as plt


GENUINE = 1
FAKE = 0

from gau import *

def lab4(data, labels):
    for idx,feature in enumerate(data):
        feature = v_row(feature)
        
        feature_genuine = feature[:,labels == GENUINE]
        feature_fake = feature[:,labels == FAKE]

        centered_feature_genuine, mean_feature_genuine = center_data(feature_genuine)
        covariance_matrix_genuine = (centered_feature_genuine @ centered_feature_genuine.T)/float(feature_genuine.shape[1])

        centered_feature_fake, mean_feature_fake = center_data(feature_fake)
        covariance_matrix_fake = (centered_feature_fake @ centered_feature_fake.T)/float(feature_fake.shape[1])

        print("Mean for feature %d and class Genuine: " % idx,mean_feature_genuine)
        print("Variance for feature %d and class Genuine:" % idx,covariance_matrix_genuine)
        print("LL for for feature %d and class Genuine: " % idx,compute_ll(feature_genuine,mean_feature_genuine,covariance_matrix_genuine) )
        print("Mean for feature %d and class Fake: " % idx,mean_feature_fake)  
        print("Variance for feature %d and class Fake: " % idx,covariance_matrix_fake)
        print("LL for for feature %d and class Fake: " % idx,compute_ll(feature_fake,mean_feature_fake,covariance_matrix_fake) )



        #print(v_col(feature).shape)
        plt.figure()
        plt.hist(feature_genuine[0], density=True, alpha = 0.5, label="Genuine", bins= 30)
        plt.hist(feature_fake[0],density=True,alpha = 0.5, label="Fake", bins=30)

        x_genuine = np.linspace(feature_genuine[0].min(), feature_genuine[0].max(), 1000)
        
        plt.plot(x_genuine.ravel(), np.exp(logpdf_GAU_ND(v_row(x_genuine), mean_feature_genuine, covariance_matrix_genuine)),label = "Genuine")

        x_fake = np.linspace(feature_fake[0].min(), feature_fake[0].max(), 1000)

        plt.title("Feature %d" % (idx + 1))
        
        plt.plot(x_fake.ravel(), np.exp(logpdf_GAU_ND(v_row(x_fake), mean_feature_fake, covariance_matrix_fake)),label = "Fake")

        plt.legend(loc = "upper right")
        plt.savefig("GAU/hists/hist" + str(idx) + ".png")
        #plt.show()





def main():
    # Open the file
    f = open("trainData.txt","r")

    # Parse the file
    data,labels = parse_file(f)

    lab4(data,labels)




if __name__ == "__main__":
    
    main()