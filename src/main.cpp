#include <iostream>
#include <Eigen\Dense>
#include <sstream>
#include <fstream>
#include <ctime>
#include "GD.h"
#include "CD.h"
#include "QN.h"
using namespace std;
using namespace Eigen;
int main()
{
	cout<<"1:abalone, 2:bodyfat, 3:housing"<<endl;
	int mode = 0;
	cin>>mode;
	string path="";
	int nums = 0;
	int features = 0; 
	switch(mode){
		case 1:
			cout<<"Using abalone dataset"<<endl;
			path="./abalone.txt";
			nums = 4177;
			features = 8;
		case 2:
			cout<<"Using bodyfat dataset"<<endl;
			path="./bodyfat.txt";
			nums = 252;
			features = 14;
			break;
		case 3:
			cout<<"Using housing dataset"<<endl;
			path="./housing.txt";
			nums = 506;
			features = 13;
			break;
		default:
			throw "Error!";
	}	 

	int w0=0;
	double lambda = 0.5;
	
	MatrixXd X(nums,features);
	MatrixXd W(features,1);
	W.fill(w0);
	MatrixXd Y(nums,1);
	ifstream dataset;
	dataset.open(path);//打开文件
	if (!dataset)
	{
		cout << "Open File Error!" << endl;
		exit(1);
	}
	
	cout<<"==Dataset loading=="<<endl;
	double x;
	for (int i=0;i<nums;i++){
		for (int j=0;j<features+1;j++){
				dataset>>x;				
				if(j==features){
					Y(i,0)=x;
					cout<<"Y"<<"("<<i<<",0)="<<x<<endl;
				}
				else{
					X(i,j)=x;
					cout<<"X"<<"("<<i<<","<<j<<")="<<x<<endl;
				}			
		}	
	}
	dataset.close();//关闭文件	
	cout<<"==Success!=="<<endl;
	
	clock_t startTime,endTime;
	startTime = clock();
	GD(Y,X,W,100,lambda);
	endTime = clock();
	cout << "The GD run time is:" <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    
	W.fill(w0);
	startTime = clock();
	CD(Y,X,W,1000,lambda);
	endTime = clock();
	cout << "The CD run time is:" <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    
	W.fill(w0);
	startTime = clock();
	QN(Y,X,W,100,lambda);
	endTime = clock();
	cout << "The QN run time is:" <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    W.fill(w0);
	cout<<"W0:"<<w0<<"Lambda:"<<lambda<<endl;


}

