#include "loadMNIST.H"
#include "matrixToFile.H"
#include "Logistic.H"
#include "Tuning.H"
#include <armadillo>
#include <iostream>

#define TUNING

using namespace arma;

int main() {
 
    /* load data */

    size_t N = 10000; // samples
    size_t M = 784; // features (without bias)
    mat X = zeros<mat>(N, M);
    vec y = zeros<vec>(N);
    loadMNIST(X, y); // already shuffled
    


    /* define model parameters */

    double alpha = 0.5;
    int niters = 1000;
    double lambda = 10;
    int nlabels = 10;
    

    
    /* write 100 images to output file for plotting */

    mat X_tofile = X.rows(0, 99);
    matrixToFile(X_tofile, "outputDigits");

    

    /* split into train and test */

    size_t nTrain = 1000;
    mat X_train = X.rows(0, nTrain-1); 
    vec y_train = y.rows(0, nTrain-1);
    mat X_test = X.rows(nTrain, N-1);
    vec y_test = y.rows(nTrain, N-1);

    

    /* initialize, train and test classifier */

    Logistic clf(alpha, niters, lambda, nlabels, "XEntropy");
    std::cout << " Training..." << std::endl;
    clf.train(X_train, y_train);
    vec predictions = clf.predict(X_test);
    //std::cout << join_horiz(predictions, y_test) << std::endl;
    double accuracyTrain = clf.score(X_train, y_train);
    double accuracyTest = clf.score(X_test, y_test);
    std::cout << "Training set accuracy is "<< accuracyTrain << std::endl;
    std::cout << "Test set accuracy is "<< accuracyTest << std::endl;
  
    
    /* output cost histories for plotting */
    
    matrixToFile(clf.getCostHistory(), "outputCostHistory");


#ifdef TUNING    

    //output learning curve data to file for plotting

    try
    {
    	learningCurves(clf, X_train, y_train, linspace<vec>(0.1, 1.0, 100), 0.8); 
    }
    catch (std::domain_error e)
    {
    	std::cout << e.what() << std::endl;
    }



    /* output cross validation data to files for plotting */
    
    try
    {
    	validationCurves(clf, X_train, y_train, linspace<vec>(0, 20, 21), 0.8, "lambda"); 
    }
    catch (std::domain_error e)
    {
    	std::cout << e.what() << std::endl;
    }


    try
    {
    	validationCurves(clf, X_train, y_train, linspace<vec>(0, 2, 21), 0.8, "alpha"); 

    }
    catch (std::domain_error e)
    {
    	std::cout << e.what() << std::endl;
    }



    
#endif

    return 0;
}



