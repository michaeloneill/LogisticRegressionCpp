#include "Logistic.H"

using namespace arma;

int main(){

    // create small test set.

    int  N = 100; // samples
    int  M = 10; // features
    int nlabels = 3; // 0, 1 and 2 (see below)
    double lambda = 0.1;
    double alpha = 0.1;
    int niters = 100;


    arma::umat y = zeros<umat>(N, 1);
    arma::mat temp = zeros<mat>(N*M, 1);

    for (int i = 0; i < N*M; ++i)
    {
	temp(i) = sin(i+1)/10;
    }

    mat X = reshape(temp, N, M);

    for (int i = 0; i < N; ++i)
    {
	y(i) = (i+1)%nlabels;
    }


    Logistic test(lambda, alpha, niters, lambda, nlabels);

    /* NB to test gradients uncomment DEBUG macro in Logistic.C */
    test.train(X, y); 

    return 0;
}
