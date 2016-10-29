CXX=g++
CXXFLAGS=-O3 -Wall -Wextra -g -I/lsc/opt/armadillo-6.100/include

Logistic.o: Logistic.C Logistic.H
	${CXX} -c $< -o $@ ${CXXFLAGS}
loadMNIST.o: loadMNIST.C loadMNIST.H
	${CXX} -c $< -o $@  ${CXXFLAGS}
matrixToFile.o: matrixToFile.C matrixToFile.H
	${CXX} -c $< -o $@  ${CXXFLAGS}
Tuning.o: Tuning.C Tuning.H
	${CXX} -c $< -o $@  ${CXXFLAGS}
main.o: main.C
	${CXX} -c $< -o $@  ${CXXFLAGS}

CXXLIBS=-L/lsc/opt/armadillo-6.100/lib -larmadillo

Logistic: Logistic.o loadMNIST.o matrixToFile.o Tuning.o main.o
	${CXX} $^ ${CXXLIBS} -o $@ ${CXXFLAGS}

clean:
	rm Logistic.o loadMNIST.o matrixToFile.o Tuning.o main.o
