#include "LogisticReg.h"
#include <direct.h>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>

/*
* ���÷���ģ��ѵ������
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
* ��������ģ��
*/
bool CLogisticReg::LogisticCreat()
{
	lrClassifier = ml::LogisticRegression::create();
	if (!lrClassifier->create())
		return false;

	return true;
}

/*
* ѵ������ģ��
* ������
* 1.ѵ������trainData
* 2.ѵ�����ݱ�ǩtrainLabel
* ���ã�
* 1.����ģ�Ͳ���
* 2.ת��ѵ����������Ϊ32λ������
* 3.ѵ��
* 4.����ģ�Ͳ�����Ϣ
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
* ����ģ��Ԥ��
* ������
* 1.��������logisFeat
* 2.������response
* 3.�������pos
* ���ã�
* 1.ת��������������Ϊ32λ������
* 2.���ط���ģ�Ͳ����ļ�
* 3.����������
* 4.Ԥ�������
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