/*
* 逻辑斯特分类器的训练与测试
*/

#ifndef TRAINING_H
#define TRAINING_H

#include "Training.h"

/*
* 根据ground-truth轨迹生成正负样本
* 参数：
* 1.轨迹集合trainTrackLet
* 2.训练数据
* 3.训练数据标签
* 作用：
* 1.在每个轨迹段中，生成相当数量正样本
* 2.在不同轨迹段中，生成相当数量负样本
*/
void GenerateLRData(vector<Tracklet> trainTrackLet, Mat &trainData, Mat &trainLabel)
{
	//generate TRUE samples
	int trueSampleCounter = 0;
	vector<Tracklet>::iterator iter;
	for (iter = trainTrackLet.begin(); iter != trainTrackLet.end(); iter++)
	{

		cout << trueSampleCounter << endl;
		
		vector<Track> tracklist = (*iter).trackInfo;
		vector<Track>::iterator tl;
		for (tl = tracklist.begin(); tl != tracklist.end() - 1; tl++)
		{
			cout << trueSampleCounter << endl;
			CLogisticFeat feat;
			char num1[100] = "";
			char num2[100] = "";
			sprintf(num1, "%.6d", (*tl).frame);
			sprintf(num2, "%.6d", (*(tl + 1)).frame);
			string frameFile = GROUND_TRUTH_IMAGE_FILE;

			Mat Img1 = imread(frameFile + num1 + ".jpg");
			Mat Img2 = imread(frameFile + num2 + ".jpg");

			Rect box1 = (*tl).box;
			Rect box2 = (*(tl + 1)).box;
			Mat tempFeat(1, 10, CV_32F);
			
			feat.TrainFeatInfo(Img1, Img2, *tl, *(tl + 1), tempFeat);

			cout << (*tl).frame << " " << (*(tl + 1)).frame << endl;
			cout << box1 << " " << box2 << endl;
			cout << tempFeat.at<float>(0, 0) << endl;
			cout << tempFeat.at<float>(0, 1) << endl;
			cout << tempFeat.at<float>(0, 2) << endl;
			cout << tempFeat.at<float>(0, 3) << endl;
			cout << tempFeat.at<float>(0, 4) << endl;
			cout << tempFeat.at<float>(0, 5) << endl;
			cout << tempFeat.at<float>(0, 6) << endl;
			cout << tempFeat.at<float>(0, 7) << endl;
			cout << tempFeat.at<float>(0, 8) << endl;
			cout << tempFeat.at<float>(0, 9) << endl;
			cout << "==========================" << endl;
			//system("pause");
			
			
			trainData.push_back(tempFeat);
			trainLabel.push_back(1.f);
			trueSampleCounter++;
			if (trueSampleCounter == POSITIVE_SAMPLE_NUM)
				break;
		}
		if (trueSampleCounter == POSITIVE_SAMPLE_NUM)
			break;
	}

	cout << "negative" << endl;

	//generate FALSE samples
	int falseSampleCounter = 0;
	vector<Tracklet>::iterator i;
	vector<Tracklet>::iterator j;
	for (i = trainTrackLet.begin(); i != trainTrackLet.end(); i++)
	{
		for (j = i + 1; j != trainTrackLet.end(); j++)
		{
			vector<Track>::iterator track_i;
			vector<Track>::iterator track_j;
			for (track_i = (*i).trackInfo.begin(); track_i != (*i).trackInfo.end(); track_i++)
			{
				for (track_j = (*j).trackInfo.begin(); track_j != (*j).trackInfo.end(); track_j++)
				{
					Rect box1, box2;
					Mat tempFeat(1, 10, CV_32F);
					box1 = (*track_i).box;
					box2 = (*track_j).box;

					float f_rectDist = sqrt(pow(box1.x - box2.x, 2) + pow(box1.y - box2.y, 2)) / (box1.height + box2.height) / 2;
					float f_euclidDist = 0.0;
					float f_histDist = compareHist((*track_i).hist, (*track_j).hist, CV_COMP_BHATTACHARYYA);

					float *pdata = tempFeat.ptr<float>(0);
					*pdata++ = (float)f_rectDist;
					*pdata++ = (float)f_euclidDist;
					*pdata++ = (float)f_histDist;
					*pdata++ = (float)(f_rectDist*f_rectDist);
					*pdata++ = (float)(f_euclidDist*f_euclidDist);
					*pdata++ = (float)(f_histDist*f_histDist);
					*pdata++ = (float)(f_rectDist*f_euclidDist);
					*pdata++ = (float)(f_rectDist*f_histDist);
					*pdata++ = (float)(f_histDist*f_euclidDist);
					*pdata++ = (float)(f_rectDist*f_euclidDist*f_histDist);

					cout << (*track_i).frame << " " << (*track_j).frame << endl;
					cout << box1 << " " << box2 << endl;
					cout << tempFeat.at<float>(0, 0) << endl;
					cout << tempFeat.at<float>(0, 1) << endl;
					cout << tempFeat.at<float>(0, 2) << endl;
					cout << tempFeat.at<float>(0, 3) << endl;
					cout << tempFeat.at<float>(0, 4) << endl;
					cout << tempFeat.at<float>(0, 5) << endl;
					cout << tempFeat.at<float>(0, 6) << endl;
					cout << tempFeat.at<float>(0, 7) << endl;
					cout << tempFeat.at<float>(0, 8) << endl;
					cout << tempFeat.at<float>(0, 9) << endl;

					trainData.push_back(tempFeat);
					trainLabel.push_back(0.f);
					falseSampleCounter++;
					if (falseSampleCounter == NEGATIVE_SAMPLE_NUM)
						break;
				}
				if (falseSampleCounter == NEGATIVE_SAMPLE_NUM)
					break;
			}
			if (falseSampleCounter == NEGATIVE_SAMPLE_NUM)
				break;
		}
		if (falseSampleCounter == NEGATIVE_SAMPLE_NUM)
			break;
	}
	system("pause");
}

/*
* 训练逻辑斯特分类器
* 1.读取ground-truth文本文件得到轨迹集合
* 2.生成正负样本
* 3.开始训练
*/
bool trainLogistic()
{
	vector<Tracklet> trackletSet;
	CLogisticReg *logisReg = new CLogisticReg;

	if (!FileToTracklet(GROUND_TRUTH_FILE, trackletSet))
		return false;

	//train logistic regress
	int startTime = GetTickCount64();
	int endTime = 0;
	cout << "Start Training...." << endl;

	Mat trainData;
	Mat trainLabel;
	GenerateLRData(trackletSet, trainData, trainLabel);

	//system("pause");

	if (!logisReg->LogisticTrain(trainData, trainLabel))
		return false;

	endTime = GetTickCount64();
	cout << "Training Stop!" << endl;
	cout << "Training Time: " << (endTime - startTime) / 1000 << "s" << endl;
}

#endif