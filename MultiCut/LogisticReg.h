/*
* 逻辑斯特分类模型的数据结构
*/

#include "DataType.h"

#ifndef _LOGISTICREG_H
#define _LOGISTICREG_H

class CLogisticReg
{
private:
	//logistic parameters
	double alpha;
	int numIters;
	int norm;			//LogisticRegression::L1 or LogisticRegression::L2.
	int trainMethod;	//LogisticRegression::BATCH or LogisticRegression::MINI_BATCH.
	int miniBatchSize;
	Ptr<ml::LogisticRegression> lrClassifier;
	bool existModel;

public:
	CLogisticReg();
	~CLogisticReg();
	bool LogisticCreat();
	bool LogisticTrain(Mat trainData, Mat trainLabel);
	bool LogisticPredict(Mat logisFeat, Mat &response, float &pos);
};


#endif