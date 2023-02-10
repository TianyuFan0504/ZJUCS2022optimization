#include <iostream>
#include <Eigen\Dense>
#include <sstream>
#include <fstream>
#include "QN.h"
#include "CD.h"
using namespace std;
using namespace Eigen;


void QN(MatrixXd Y,MatrixXd X,MatrixXd W,int iteration,double lambda){
	cout<<"using Quasi-Newton to solve"<<endl;
	int nums = X.rows();
	int features = X.cols();
	MatrixXd MSELoss(1,1);
	MatrixXd g_now(features,1);
	MatrixXd g_last(features,1);
	MatrixXd dg(features,1);
	MatrixXd dW(features,1);
	MatrixXd d(features,1);
	MatrixXd eye;
	MatrixXd B;
	eye = MatrixXd::Identity(features, features); 
	B = MatrixXd::Identity(features, features); 
//	MatrixXd B(features,features);
//	B.setIdentity(features, features);
	double L;
	double a;
	double beta;
	ofstream dataFile;
	dataFile.open("./QN.txt", ofstream::app);
	fstream file("./QN.txt", ios::out);
	int count_flag=0;
	for(int i=0;i<iteration;i++){
		cout<<"Epoch:"<<i<<" ";
		MSELoss = (1.0/nums) * (( Y - X * W ).transpose()) * ( Y - X * W);
		L = MSELoss(0,0);
		cout<<"Loss:"<<L<<endl;
		dataFile <<L<< endl;     // 写入数据
		
		
		if(i==0){
			g_now = (-2.0/nums) * X.transpose() * ( Y - X * W ) +lambda*  W;
		}

		d = -B * g_now;
		
		
		a = Linear_search(Y,X,W,d,lambda);
//		cout<<"a:"<<a<<endl;
		W = W + a * d;
//		cout<<"W:"<<W.transpose()<<endl;
		if(a == 0.){
			count_flag +=1;

		}
		else{
			count_flag=0;
			
		}
		if(count_flag ==10){
			cout<<"Early stop!"<<endl;
			break;
		}
		g_last = g_now;
		g_now = (-2.0/nums) * X.transpose() * ( Y - X * W ) +lambda*  W;
		
		dg = g_now-g_last;
		dW = a * d;
		
		B = (eye - ( (dW * dg.transpose()) / ((dW.transpose() * dg)(0,0) +1e-5)) ) * B * (eye - ( (dg * dW.transpose()) / ((dW.transpose() * dg)(0,0)+1e-5) )) +( (dW * dW.transpose()) / ((dW.transpose() * dg)(0,0)+1e-5) ); 
//		cout<<"B:"<<B<<endl;
		


	}
	dataFile.close();                           // 关闭文档
	
	
		
	
}




