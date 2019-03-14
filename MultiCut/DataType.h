#pragma warning(disable:4996)
/*
* 数据结构：
* 1.检测框Track：帧数frame，检测框位置box，id（初始化-1，表示该检测框为未跟踪状态），hist（颜色直方图信息）
* 2.轨迹Tracklet：出现的第一帧firstFrame，出现的最后一帧lastFrame，id，不再与其他轨迹片段连接时为true，跟踪到的每个检测框的详细信息trackInfo
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