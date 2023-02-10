#include <iostream>
#include <Eigen\Dense>
#include <sstream>
#include <fstream>
#include "GD.h"
using namespace std;
using namespace Eigen;


void GD(MatrixXd Y,MatrixXd X,MatrixXd W,int iteration,double lambda){
	cout<<"using Gradient Descent to solve"<<endl;
	int nums = X.rows();
	int features = X.cols();
	MatrixXd MSELoss(1,1);
	MatrixXd WGrad(features,1);
	double L;
	ofstream dataFile;
	dataFile.open("./GD.txt", ofstream::app);
	fstream file("./GD.txt", ios::out);
	double lr = 0.01;
	for(int i=0;i<iteration;i++){
		cout<<"Epoch:"<<i<<" ";
		MSELoss = (1.0/nums) * (( Y - X * W ).transpose()) * ( Y - X * W);
		L = MSELoss(0,0);
		cout<<"Loss:"<<L<<endl;
		dataFile <<L<< endl;     // 写入数据
		WGrad = (-2.0/nums) * X.transpose() * ( Y - X * W ) +lambda*  W;
//		cout<<"Grad:"<<WGrad.transpose()<<endl;
		W = W-lr*WGrad;
//		cout<<"W:"<<W.transpose()<<endl;

	}
	dataFile.close();                           // 关闭文档
}

 
