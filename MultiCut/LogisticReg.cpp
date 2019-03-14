#include "LogisticReg.h"
#include <direct.h>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>

/*
* 设置分类模型训练参数
*/
CLogisticReg::CLogisticReg()
{
	alpha = 0.05;
	numIters = 100;
	norm = ml::LogisticRegression::REG_L2;
	trainMethod = ml::LogisticRegression::MINI_BATCH;
	miniBatchSize = 10;
	
}

CLogisticReg::~CLogisticReg()
{

}

/*
* 建立分类模型
*/
bool CLogisticReg::LogisticCreat()
{
	lrClassifier = ml::LogisticRegression::create();
	if (!lrClassifier->create())
		return false;

	return true;
}

/*
* 训练分类模型
* 参数：
* 1.训练数据trainData
* 2.训练数据标签trainLabel
* 作用：
* 1.设置模型参数
* 2.转换训练数据类型为32位浮点型
* 3.训练
* 4.保存模型参数信息
*/
bool CLogisticReg::LogisticTrain(Mat trainData, Mat trainLabel)
{
	if (!LogisticCreat())
		return false;

	lrClassifier->setLearningRate(alpha);
	lrClassifier->setIterations(numIters);
	lrClassifier->setRegularization(norm);
	lrClassifier->setTrainMethod(trainMethod);
	lrClassifier->setMiniBatchSize(miniBatchSize);

	trainData.convertTo(trainData, CV_32F);
	trainLabel.convertTo(trainLabel, CV_32F);

	if (!lrClassifier->train(trainData, ml::ROW_SAMPLE, trainLabel))
		return false;

	lrClassifier->save(CLASSIFIER_MODEL_FILE);

	return true;
}

/*
* 分类模型预测
* 参数：
* 1.测试数据logisFeat
* 2.分类结果response
* 3.分类概率pos
* 作用：
* 1.转换测试数据类型为32位浮点型
* 2.加载分类模型参数文件
* 3.计算分类概率
* 4.预测分类结果
*/
bool CLogisticReg::LogisticPredict(Mat logisFeat, Mat &response, float &pos)
{
	Mat result(1, 1, CV_32S);
	logisFeat.convertTo(logisFeat, CV_32F);

	if (!lrClassifier)
		lrClassifier = ml::LogisticRegression::load<ml::LogisticRegression>(CLASSIFIER_MODEL_FILE);

	Mat theta = lrClassifier->get_learnt_thetas();

	Mat x = Mat::ones(1, 11, CV_32F);
	logisFeat.copyTo(x.colRange(1, 11));

	//Mat x = Mat::ones(1, 6, CV_32F);
	//logisFeat.copyTo(x.colRange(1, 6));

	Mat thetaX = x*theta.t();
	Mat expItem;
	Mat g(1, 1, CV_32F);
	exp(-thetaX, expItem);
	g = 1.0 / (1 + expItem);
	pos = g.at<float>(0, 0);

	lrClassifier->predict(logisFeat, result);
	result.copyTo(response);
	result.release();
	return true;
}