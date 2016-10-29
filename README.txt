The source code files in miniProject1Logistic.tar.gz relate to the logistic classifier model in the mini-project report "Linear classification models applied to the classification of hand-written digits".

The MNIST data files should be downloaded from http://cis.jhu.edu/~sachin/digit/digit.html and organised by digit into a folder MNISTdata, with MNISTdata/data0 corresponding to digit 0 etc. all the way up to digit 9.

The project uses the c++ armadillo linear algebra library. The included Makefile has the required linker flags to this library for running the code on the LSC computers. 

The results relating to the logistic classifier in the aforementioned report can be re-produced by varying the parameters in main.C, which have been labelled with self-explanatory names.

To compile the program:

make Logistic 

To run it:

./Logistic

Numerical gradient checking can be performed by toggling on/off the DEBUG_GRADIENTS macro in Logistic.C. 

The file plotLogistic.py can be used to generate all the plots. Note that output data files for the plotting of learning curves and validation curves will only be produced if the TUNING macro in main.C is set to true.

