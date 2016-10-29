#ifndef LOGISTIC_H
#define LOGISTIC_H

#include "BaseClassifier.H"
#include <armadillo>

#include <string>

class Logistic: public BaseClassifier {
public:

    Logistic(double, int, double lambda = 0, int nlabels = 0, std::string costType="XEntropy");
    void train(arma::mat&, arma::vec&);
    arma::vec predict(const arma::mat&);
    double score(const arma::mat&, const arma::vec&);
    
    /* setters/getters */

    void setLambda(double lambda) { mLambda = lambda; }
    void setAlpha(double alpha) { mAlpha = alpha; }
    const arma::mat& getParams() const { return mParams;}
    const arma::mat& getCostHistory() const { return mCostHistory;}
    
 
private:
  
    arma::mat mX, mParams, mCostHistory, mGrad;
    arma::vec my;

    double mAlpha;
    int mIters;
    double mLambda;
    int mLabels;
    std::string mCostType;

    void gradientDescent();
    arma::mat grad();
    void checkGradients();
    arma::rowvec cost(const arma::mat&);

    arma::umat yToFull(const arma::vec&);
};

// non-members

arma::mat sigmoid(const arma::mat&);
arma::mat sigmoidGrad(const arma::mat&);

#endif
