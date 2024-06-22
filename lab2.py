

GENUINE = 1
FAKE = 0

from utils import center_data, compute_var_std
import matplotlib.pyplot as plt


def show_scatter(genuine,fake):
    # For each feature
    for feature1 in range(6):
        for feature2 in range(6):
            if feature1 == feature2:
                continue
            # Create the scatter plot
            plt.figure()
            plt.scatter(genuine[feature1],genuine[feature2],alpha=0.7 ,label= "Genuine")
            plt.scatter(fake[feature1],fake[feature2], alpha=0.7, label= "Fake")

            plt.legend(loc = "upper right")
            plt.xlabel("Feature " + str(feature1))
            plt.ylabel("Feature " + str(feature2))
            plt.title("Feature %d vs Feature %d" % (feature1,feature2))
            plt.savefig('./scatter/scatter_%d_%d.png' % (feature1, feature2))
        #plt.show()


    return

def show_hists(genuine,fake):
    # For each feature
    for feature in range(6):
        # Create the histogram for the selected feature
        plt.figure()
        plt.hist(genuine[feature], density=True, alpha = 0.5, label= "Genuine",bins= 30)
        plt.hist(fake[feature], density=True, alpha = 0.5, label= "Fake",bins= 30)

        plt.legend(loc = "upper right")
        plt.xlabel("Feature " + str(feature))
        plt.title("Feature %d" % (feature + 1))
        plt.savefig('./histograms/hist_%d.png' % feature)

def lab2(data,labels):
    # Split by class
    genuine = data[:,labels == GENUINE]
    fake = data[:,labels == FAKE]

    # Compute the mean for the genuine class and center the data
    genuine_centered, mean_genuine = center_data(genuine)

    print("Mean genuine class: \n", mean_genuine)

    # Compute the mean for the fake class and center the data
    fake_centered, mean_fake = center_data(fake)

    print("Mean fake class: \n", mean_fake)

    # Get the variance and standard daviation for genuine class
    var_genuine,std_genuine = compute_var_std(genuine_centered)

    print("Variance for genuine: \n",var_genuine,"\nStandard deviation for genuine: \n",std_genuine)

    # Get the variance and standard daviation for fake class
    var_fake,std_fake = compute_var_std(fake_centered)

    print("Variance for fake: \n",var_fake,"\nStandard deviation for fake: \n",std_fake)

    show_hists(genuine,fake)
    
    #show_scatter(genuine,fake)

    show_hists(genuine_centered,fake_centered)

    #show_scatter(genuine_centered,fake_centered)




from utils import parse_file

def main():
    # Open the file
    f = open("trainData.txt","r")

    # Parse the file
    data,labels = parse_file(f)

    lab2(data,labels)




if __name__ == "__main__":

    main()