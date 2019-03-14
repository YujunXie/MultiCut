#pragma once
/*
* ����ģ�����ݽṹ
*/

#include "DataType.h"

#ifndef _LOGISRIC_FEAT_H
#define _LOGISRIC_FEAT_H

class CLogisticFeat
{
private:
	Rect r_prev;
	Rect r_next;
	double d_histDist;
	double d_euclidDist;

	/*����������ý������Ĳ���*/
	const double	OPTICAL_FLOW_PYRAMID_SCALE = 0.4;
	const int		OPTICAL_FOLW_PYRAMID_LEVELS = 1;
	const int		OPTICAL_FLOW_PYRAMID_WINDOWSIZE = 12;
	const int		OPTICAL_FLOW_PYRAMID_ITERATIONS = 2;
	const int		OPTICAL_FLOW_NEIGHBOR = 7;
	const double	OPTICAL_FLOW_GAUSSIAN_SIGMA = 1.2;

	/*�����ֿ������С*/
	const int		GRID_SIZE = 5;

	/*ŷʽ�����˹�任����*/
	const int		NORMALIZE_GAUSSIAN_SIGMA = 7;

	bool TrainLogisticFeat(Mat Img_prev, Mat Img_next, Rect r_prev, Rect r_next);
	bool CLogisticFeat::TestLogisticFeat(Mat opticalFlow, Mat Img_prev, Rect r_prev, Rect r_next);

public:
	Mat m_opticalFlow;
	CLogisticFeat();
	~CLogisticFeat();
	bool GetOpticalFlowMap(Mat Img_prev, Mat Img_next);
	bool TrainFeatInfo(Mat Img_prev, Mat Img_next, Track prev, Track next, Mat &feature);
	bool CLogisticFeat::TestFeatInfo(Mat opticalFLow, Mat Img_prev, Track prev, Track next, Mat &feature);
	
};


#endif