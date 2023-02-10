#include <iostream>
#include <Eigen\Dense>
#include <sstream>
#include <fstream>
#include "CD.h"
using namespace std;
using namespace Eigen;

double Linear_search(MatrixXd Y,MatrixXd X,MatrixXd W_input,MatrixXd d,double lambda){
	double a = -1;
	double min =1e5;
	double output = 0.;
	double temp_loss =0 ;
	int nums = X.rows();
	int features = X.cols();
	MatrixXd RRLoss(1,1);
	MatrixXd W(features,1);
	for (a=0;a<1;a+=0.001){
		W = W_input+a*d;
		RRLoss = (1.0/nums) * (( Y - X * W ).transpose()) * ( Y - X * W)+lambda* W.transpose() * W;
		temp_loss=RRLoss(0,0);
//		cout<<"temp_loss:"<<temp_loss<<endl;
		if (min>temp_loss){
			min=temp_loss;
			output=a;	
		}
	}
	return output;
	
	
}


void CD(MatrixXd Y,MatrixXd X,MatrixXd W,int iteration,double lambda){
	cout<<"using Conjugate Descent to solve"<<endl;
	int nums = X.rows();
	int features = X.cols();
	MatrixXd MSELoss(1,1);
	MatrixXd g_now(features,1);
	MatrixXd g_last(features,1);
	MatrixXd d(features,1);
	double L;
	double a;
	double beta;
	ofstream dataFile;
	dataFile.open("./CD.txt", ofstream::app);
	fstream file("./CD.txt", ios::out);
	int count_flag =0;
	for(int i=0;i<iteration;i++){
		cout<<"Epoch:"<<i<<" ";
		MSELoss = (1.0/nums) * (( Y - X * W ).transpose()) * ( Y - X * W);
		L = MSELoss(0,0);
		cout<<"Loss:"<<L<<endl;
		dataFile <<L<< endl;     // 写入数据
		
		
		g_now = (-2.0/nums) * X.transpose() * ( Y - X * W ) +lambda*  W;
		
		if(i==0){
			d = -g_now;
		}
		
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
		if(count_flag ==5){
			cout<<"Early stop!"<<endl;
			break;
		}
		g_last = g_now;
		g_now = (-2.0/nums) * X.transpose() * ( Y - X * W ) +lambda*  W;
		
		beta = g_now.norm()*1.0/g_last.norm();
		
		d = -g_now + beta * d;	
		

	}
	dataFile.close();                           // 关闭文档
	
	
		
	
}




