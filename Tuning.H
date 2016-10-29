#ifndef DEBUGGING_H
#define DEBUGGING_H

#include "BaseClassifier.H"
#include <armadillo>

#include <string>

void learningCurves(BaseClassifier&, arma::mat&, arma::vec&, const arma::vec, const double);

void validationCurves(BaseClassifier&, arma::mat&, arma::vec&, const arma::vec, const double, std::string);

#endif
