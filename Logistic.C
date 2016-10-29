#include "Logistic.H"
#include "matrixToFile.H"
#include <armadillo>
#include <iostream>
#include <cassert>
#include <stdexcept>

//#define DEBUG_GRADIENTS //comment out when happy

using namespace arma;
using std::string;

Logistic::Logistic(double alpha, int niters, double lambda, int nlabels, string costType): mAlpha(alpha), mIters(niters), mLambda(lambda), mLabels(nlabels){ 

    if (costType == "XEntropy" | costType == "Quadratic")
    {
	mCostType = costType;
    }
    else
    {
	throw std::domain_error("Invalid Cost Type");
    }

#ifdef DEBUG_GRADIENTS

    // create small test set
    
    int N = 10; // no. of samples
    int M = 20; // dimension of samples
    vec XUnrolled = zeros(N*M);
    
    for (int i = 0; i != N*M; ++i)
    {
	XUnrolled(i) = sin(i+1)/10;
    }
    
    mX = reshape(XUnrolled, N, M);
    my = zeros(N);
    
    for (int i = 0; i != N; ++i)
    {
	my(i) = (i+1)%nlabels;
    }

    mParams = randu<mat>(M, mLabels);
    
    checkGradients(); // simultaneously checks gradients for each class
    
#endif
    
}



void Logistic::train(mat& X, vec& y){

    assert(y.n_rows == X.n_rows);
    mX = join_horiz(ones<mat>(X.n_rows), X); // add bias
    my = y;
  
    /* Initialise/reset remaining members. To be filled after training */
    mParams = zeros<mat>(mX.n_cols, mLabels); 
    mGrad = zeros<mat>(mX.n_cols, mLabels); 
    mCostHistory = zeros<mat>(mIters, mLabels);


    /* Simultaneous one-vs-rest */

    gradientDescent();

}



void Logistic::gradientDescent(){

    for (int iter = 0; iter < mIters; ++iter)
    {
	mCostHistory.row(iter) = cost(mParams); // store cost for each classifier
	mGrad = grad(); // update gradient for each classifier 
	mParams = mParams - mAlpha*mGrad; // update parameters for each classifier
	if ((iter+1)%100 == 0)
	{
	    std::cout << (iter+1) << " iterations completed" << std::endl;
	}
    }
}



vec Logistic::predict(const mat& X){

    assert(X.n_cols == mX.n_cols - 1);
    
    mat X_test = join_horiz(ones<mat>(X.n_rows), X);

    /* find class confidences */

    mat confidences = sigmoid(X_test*mParams);
    uvec predictions = zeros<uvec>(X_test.n_rows);

    for (size_t i = 0; i != confidences.n_rows; ++i)
    {
	/* index of max confidence for each sample is class identifier. */
	/* store this in uword predictions. */

	confidences.row(i).max(predictions(i)); 

    }

    return conv_to<vec>::from(predictions); 
}



double Logistic::score(const mat& X, const vec& y){

    size_t n = X.n_rows;
    assert(n == y.n_rows);
    
    vec predictions = predict(X);
    return accu(predictions == y)/(double)n;
    
}


mat Logistic::grad(){

    /* Simultaneously computes gradients for each of K classifiers */
    
    size_t N = mX.n_rows;
    mat Z = mX*mParams; // NxK
    umat yFull = yToFull(my); // NxK

    mat temp = mParams;
    temp.row(0) = zeros<rowvec>(mLabels); // no regularisation for first element of each grad
    mat g = zeros(size(mParams));
    if (mCostType == "XEntropy")
    {
	g = 1/(double)N * mX.t()*(sigmoid(Z) - yFull) + (mLambda/(double)N)*temp;
    }
    else // must be quadratic 
    {
	g = 1/(double)N * mX.t()*((sigmoid(Z) - yFull)%sigmoidGrad(Z)) + (mLambda/(double)N)*temp;
    }	
    return g;
}



void Logistic::checkGradients(){

    /* simultaneously checks gradients against numerical approx for all classes */

    int M = mParams.n_rows;
    mat numgrad = zeros<mat>(M, mLabels); //MxK
    vec perturbed = zeros<vec>(M);
    double delta = 1e-4;

    
    for (int i = 0; i < M; ++i)
    {
	perturbed(i) = delta;
	mat theta1 = mParams.each_col() - perturbed; //element-wise
	mat theta2 = mParams.each_col() + perturbed; //element-wise

	rowvec cost1 = cost(theta1);
	rowvec cost2 = cost(theta2);
	
	numgrad.row(i) = (cost2 - cost1)/(2*delta);
	perturbed(i) = 0; // reset for next round
    }

    mat g = grad(); // MxK

    for (int c = 0; c < mLabels; ++c)
    {
	std::cout << "Gradient check for class " << c << endl;
	std::cout << join_horiz(numgrad.col(c), g.col(c)) << std::endl;
    }

}



rowvec Logistic::cost(const mat& theta){

    size_t N = mX.n_rows;
    size_t M = mX.n_cols;

    assert(theta.n_rows == M);
    
    mat Z = mX*theta; // NxK

    umat yFull = yToFull(my); // NxK

    /* convert  umat to mat so can negate */

    mat Y = conv_to<mat>::from(yFull);

    /* note no regularisation for bias. */

    rowvec J = zeros<rowvec>(mLabels);
    
    if (mCostType == "XEntropy")
    {
	J = 1/(double)N * sum(-Y%log(sigmoid(Z)) - (1 - Y)%log(1 - sigmoid(Z))) + mLambda/(2*(double)N) * sum(theta.rows(1, M-1)%theta.rows(1, M-1)); 
    }
    else // it must be quadratic 
    {
	J = 1/(2*(double)N) *sum((sigmoid(Z) - Y)%(sigmoid(Z) - Y)) + mLambda/(2*(double)N) * sum(theta.rows(1, M-1)%theta.rows(1, M-1)); 
	
    }
    return J;
}



umat Logistic::yToFull(const vec& y){

    uword N = y.n_rows;
    umat yFull = zeros<umat>(N, mLabels);
    
    for (int c = 0; c != mLabels; ++c)
    {
	yFull.col(c) = (y==c);
    }

    return yFull;
}



/* non-members */

mat sigmoid(const mat& z){

    mat g = zeros(size(z));
    g = 1/(1+exp(-z));
    return g;

}

mat sigmoidGrad(const mat& z){
    
    return sigmoid(z)%(1-sigmoid(z));

}
