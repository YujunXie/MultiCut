#pragma warning(disable:4996)
/*
* ���ݽṹ��
* 1.����Track��֡��frame������λ��box��id����ʼ��-1����ʾ�ü���Ϊδ����״̬����hist����ɫֱ��ͼ��Ϣ��
* 2.�켣Tracklet�����ֵĵ�һ֡firstFrame�����ֵ����һ֡lastFrame��id�������������켣Ƭ������ʱΪtrue�����ٵ���ÿ���������ϸ��ϢtrackInfo
*/

#ifndef DATATYPE_H
#define DATATYPE_H

#include <cstdio>
#include <ctime>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <Windows.h>
#include <opencv.hpp>
#include <opencv2\opencv.hpp>
#include "andres/graph/graph.hxx"
#include "andres/graph/complete-graph.hxx"
#include "andres/graph/multicut/kernighan-lin.hxx"
#include "andres/graph/multicut-lifted/kernighan-lin.hxx"

using namespace std;
using namespace cv;

#define GROUND_TRUTH_IMAGE_FILE "E:/PAPER/Code/source/MOT16/MOT16-13/img1/"
#define GROUND_TRUTH_FILE "E:/PAPER/Code/source/MOT16/MOT16-13/gt/gt.txt"
#define CLASSIFIER_MODEL_FILE "E:/PAPER/Code/Test/MultiCut/Data/MOT16-13/logistic_reg_model_13.xml"
#define DETECTION_FILE "E:/PAPER/Code/source/MOT16/MOT16-13/det/det.txt"
#define VISIUAL_RESULT_FILE "E:/PAPER/Code/Test/MultiCut/Data/MOT16-13/result.txt"

#define POSITIVE_SAMPLE_NUM 750  //1000
#define NEGATIVE_SAMPLE_NUM 500 //200

#define FRAMENUM 5

struct Track
{
	int personId;
	int frame;
	Rect box;
	Mat hist;
};

struct Tracklet
{
	int personId;
	int firstFrame;
	int lastFrame;
	bool isEnd;
	vector<Track> trackInfo;
};

#endif