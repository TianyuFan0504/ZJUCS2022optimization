#pragma once
#include <iostream>
#include <Eigen\Dense>
#include <sstream>
#include <fstream>
using namespace std;
using namespace Eigen;
void CD(MatrixXd Y,MatrixXd X,MatrixXd W,int iteration,double lambda);
double Linear_search(MatrixXd Y,MatrixXd X,MatrixXd W_input,MatrixXd d,double lambda);
