#pragma once
#include <iostream>
#include <Eigen\Dense>
#include <sstream>
#include <fstream>
using namespace std;
using namespace Eigen;
void GD(MatrixXd Y,MatrixXd X,MatrixXd W,int iteration,double lambda);
