#ifndef BASECLASSIFIER_H
#define BASECLASSIFIER_H

#include <armadillo>

class BaseClassifier {
public:

    virtual ~BaseClassifier(){}
    virtual void setLambda(double) = 0;
    virtual void setAlpha(double) = 0;
    virtual void train(arma::mat&, arma::vec&) = 0;
    virtual arma::vec predict(const arma::mat&) = 0;
    virtual double score(const arma::mat&, const arma::vec&) = 0;
    
    
};

#endif
