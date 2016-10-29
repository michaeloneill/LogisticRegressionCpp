#include "Tuning.H"
#include "matrixToFile.H"

/* Polymorphism enabled by passing by reference rather than pointer */
/* This means function can't accept NULL objects */

using namespace arma;
using std::string;

void learningCurves(BaseClassifier& clf, mat& X, vec& y, const vec batches, const double split){

    std::cout << "computing learning curve data " << std::flush;

    int N = X.n_rows;
    int nBatches = batches.n_elem;

    mat score_train = zeros<mat>(nBatches);
    mat score_val = zeros<mat>(nBatches);

    if (split <= 1.0 && split > 0)
    {
	/* Use some of training data for validation */
	
	int nTrain = ceil(split*N);
	mat X_train = X.rows(0, nTrain-1);
	vec y_train = y.rows(0, nTrain-1);
	mat X_val = X.rows(nTrain, N-1);
	vec y_val = y.rows(nTrain, N-1);
	
	for (int i = 0; i != nBatches; ++i)
	{
	    std::cout << "Learning Curves: Training batch " << i+1 << std::endl;
	    
	    if (batches(i) <= 1.0 && batches(i) > 0)
	    {
		int batchSize = ceil(batches(i)*X_train.n_rows);
		mat X_batch = X_train.rows(0, batchSize-1);
		vec y_batch = y_train.rows(0, batchSize-1);
	   
		clf.train(X_batch, y_batch);
		
		score_train(i) = clf.score(X_batch, y_batch);
		score_val(i) = clf.score(X_val, y_val);
	    }
	    else
	    {
		throw std::domain_error("batches elements must be (0, 1]");

	    }
	}
    }
    else
    {
	throw std::domain_error("split must be (0, 1]");
    }
    
    mat scores = join_horiz(score_train, score_val);
    scores = join_horiz(conv_to<mat>::from(batches), scores);
    matrixToFile(scores, "outputLogisticLC");
    std::cout << std::endl;
}

/* Polymorphism enabled by passing by reference rather than pointer */
/* This means function can't accept NULL objects */

void validationCurves(BaseClassifier& clf, mat& X, vec& y, const vec values, const double split, string valType){


    if ((valType != "lambda") && (valType != "alpha"))
    {
	throw std::domain_error("Invalid validation parameter type");
    }		

    std::cout << "computing validation data " << std::flush;
    int N = X.n_rows;
    int nValues = values.n_elem;

    mat score_train = zeros<mat>(nValues);
    mat score_val = zeros<mat>(nValues);

    if (split <= 1.0 && split > 0)
    {
	/* Use some of training data for validation */
	
	int nTrain = ceil(split*N);
	mat X_train = X.rows(0, nTrain-1);
	vec y_train = y.rows(0, nTrain-1);
	mat X_val = X.rows(nTrain, N-1);
	vec y_val = y.rows(nTrain, N-1);
	
	for (int i = 0; i != nValues; ++i)
	{
	    if (values(i) >= 0)
	    {
		if (valType == "lambda")
		{
		    std::cout << "Lambda Validation Curves: Training for lambda = " << values(i) << std::endl;
		    clf.setLambda(values(i));
		}
		else // must be alpha
		{
		    std::cout << "Alpha Validation Curves: Training for alpha = " << values(i) << std::endl;
		    clf.setAlpha(values(i));
		}
		
		clf.train(X_train, y_train);
		score_train(i) = clf.score(X_train, y_train);
		score_val(i) = clf.score(X_val, y_val);
	    }
	    else
	    {
		throw std::domain_error("values must be >= 0");

	    }
	}
    }
   
    else
    {
	throw std::domain_error("split must be (0, 1]");
    }

    mat scores = join_horiz(score_train, score_val);
    scores = join_horiz(conv_to<mat>::from(values), scores);
    if (valType == "lambda")
    {
	matrixToFile(scores, "outputLogisticLamVal");
    }
    else //must be alpha
    {
	matrixToFile(scores, "outputLogisticAlphaVal");
    }
    std::cout << std::endl;
}
