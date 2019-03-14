/*
* 特征模型具体操作
* 1.特征A：检测框大小相似度LogisticFeat
* 2.特征B：表观模型
* 3.特征C：运动模型
*/

#include "LogisticFeat.h"

CLogisticFeat::CLogisticFeat()
{

}

CLogisticFeat::~CLogisticFeat()
{

}

/*
* 获取光流图像
* 参数:
* 1.前一帧图像Img_prev
* 2.后一帧图像Img_next
* 作用：
* 1.将彩色图转换成灰度图
* 2.计算稠密光流
* 3.计算成功直接释放原先的光流数据
*/
bool CLogisticFeat::GetOpticalFlowMap(Mat Img_prev, Mat Img_next)
{
	if (Img_prev.empty() || Img_next.empty())
		cout << "输入了空图像" << endl;
	Mat Img_prevGray, Img_nextGray;
	if (Img_prev.channels() == 1)
		Img_prev.copyTo(Img_prevGray);
	else
		cvtColor(Img_prev, Img_prevGray, COLOR_BGR2GRAY);
	if (Img_next.channels() == 1)
		Img_next.copyTo(Img_nextGray);
	else
		cvtColor(Img_next, Img_nextGray, COLOR_BGR2GRAY);

	Mat tempFlow;
	calcOpticalFlowFarneback(Img_prevGray, Img_nextGray, tempFlow,
		OPTICAL_FLOW_PYRAMID_SCALE, OPTICAL_FOLW_PYRAMID_LEVELS,
		OPTICAL_FLOW_PYRAMID_WINDOWSIZE, OPTICAL_FLOW_PYRAMID_ITERATIONS,
		OPTICAL_FLOW_NEIGHBOR, OPTICAL_FLOW_GAUSSIAN_SIGMA,
		0);
	if (tempFlow.empty())
	{
		cout << "计算光流失败" << endl;
		tempFlow.release();
		return false;
	}
	else
	{
		m_opticalFlow.release();
		tempFlow.copyTo(m_opticalFlow);
		tempFlow.release();
		return true;
	}
}

/*
* 训练过程中获取光流距离
* 参数:
* 1.前一帧图像Img_prev
* 2.后一帧图像Img_next
* 3.前一帧中的检测框r_prev
* 4.后一帧中的检测框r_next
* 作用：
* 1.获取光流图像
* 2.根据光流计算Img_next中r_prev的中心位置
* 3.计算r_prevCenterPredict与r_nextCenter的欧式距离并高斯化
*/
bool CLogisticFeat::TrainLogisticFeat(Mat Img_prev, Mat Img_next, Rect r_prev, Rect r_next)
{
	int imgRow = Img_prev.rows;
	int imgCol = Img_prev.cols;

	if (!GetOpticalFlowMap(Img_prev, Img_next))
	{
		d_euclidDist = -1;
		return false;
	}

	Rect rectNormal = r_prev;
	if (rectNormal.x < 0)
	{
		rectNormal.width -= rectNormal.x;
		rectNormal.x = 0;
	}
	if (rectNormal.y < 0)
	{
		rectNormal.height -= rectNormal.y;
		rectNormal.y = 0;
	}
	if ((rectNormal.y + rectNormal.height) > Img_prev.rows)
		rectNormal.height = Img_prev.rows - rectNormal.y;
	if ((rectNormal.x + rectNormal.width) > Img_prev.cols)
		rectNormal.width = Img_prev.cols - rectNormal.x;


	//得到r_prev的中心点
	Point2f r_prevCenter((rectNormal.tl() + rectNormal.br()) / 2);
	//处于boundingbox中的光流
	Mat flowInBox(m_opticalFlow(rectNormal));
	Mat r_prevCenterMotionBoard = Mat::zeros(Img_prev.size() / GRID_SIZE, CV_32SC1);

	for (int y = 0; y < flowInBox.rows; y++)
	{
		for (int x = 0; x < flowInBox.cols; x++)
		{
			//各个像素的光流
			Point2f r_prevCenterMovingDir = flowInBox.at<Point2f>(y, x);
			//box中心点按光流运动到新位置
			Point2f r_prevNewCenter = r_prevCenter + r_prevCenterMovingDir;
																  
			int moveRow = cvRound(r_prevNewCenter.y / GRID_SIZE);
			int moveCol = cvRound(r_prevNewCenter.x / GRID_SIZE);
			moveRow = max(moveRow, 0);
			moveCol = max(moveCol, 0);
			moveRow = min<int>(moveRow, r_prevCenterMotionBoard.rows - 1);
			moveCol = min<int>(moveCol, r_prevCenterMotionBoard.cols - 1);

			//r_prevNewCenter.x = cvRound(r_prevNewCenter.x);
			//r_prevNewCenter.y = cvRound(r_prevNewCenter.y);
			/*int moveRow = cvRound((y + r_prevNewCenter.y) / GRID_SIZE);
			int moveCol = cvRound((x + r_prevNewCenter.x) / GRID_SIZE);*/
			//moveRow = min(moveRow, cvRound(imgRow / GRID_SIZE));
			//moveCol = min(moveCol, cvRound(imgCol / GRID_SIZE));

			//预测位置记录加1
			++r_prevCenterMotionBoard.at<int>(moveRow, moveCol);
		}
	}

	Point r_prevCenterMostLikelyLoaction, useless;
	double useless1, useless2;
	minMaxLoc(r_prevCenterMotionBoard, &useless1, &useless2, &useless, &r_prevCenterMostLikelyLoaction);
	Point2f r_prevCenterPredict = r_prevCenterMostLikelyLoaction * GRID_SIZE;

	//得到原始box的运动预测位置
	r_prevCenterPredict.x += GRID_SIZE / 2;
	r_prevCenterPredict.y += GRID_SIZE / 2;

	Point2f r_nextCenter((r_next.tl() + r_next.br()) / 2);
	d_euclidDist = sqrt(pow(r_prevCenterPredict.x - r_nextCenter.x, 2) + pow(r_prevCenterPredict.y - r_nextCenter.y, 2));
	//欧式距离高斯化
	d_euclidDist = exp(-pow(d_euclidDist, 2) / pow(7, 2));

	flowInBox.release();
	r_prevCenterMotionBoard.release();

	return true;
}

/*
* 训练过程中获取特征信息
* 参数:
* 1.前一帧图像Img_prev
* 2.后一帧图像Img_next
* 3.前一帧中的检测框r_prev
* 4.后一帧中的检测框r_next
* 5.特征信息矩阵feature
* 作用：
* 1.获取光流距离特征
* 2.获取矩形框大小距离特征
* 3.获取颜色直方图距离特征
*/
bool CLogisticFeat::TrainFeatInfo(Mat Img_prev, Mat Img_next, Track prev, Track next, Mat &feature)
{
	if (!TrainLogisticFeat(Img_prev, Img_next, prev.box, next.box))
	{
		cout << "距离特征计算失败" << endl;
		return false;
	}

	feature.release();
	feature = Mat::zeros(1, 10, CV_32FC1);
	
	double d_rectDist = sqrt(pow(prev.box.x - next.box.x, 2) + pow(prev.box.y - next.box.y, 2)) / (prev.box.height / 2 + next.box.height / 2);
	d_histDist = compareHist(prev.hist, next.hist, CV_COMP_BHATTACHARYYA);

	float *pdata = feature.ptr<float>(0);
	*pdata++ = (float)d_rectDist;
	*pdata++ = (float)d_euclidDist;
	*pdata++ = (float)d_histDist;
	*pdata++ = (float)(d_rectDist*d_rectDist);
	*pdata++ = (float)(d_euclidDist*d_euclidDist);
	*pdata++ = (float)(d_histDist*d_histDist);
	*pdata++ = (float)(d_rectDist*d_euclidDist);
	*pdata++ = (float)(d_rectDist*d_histDist);
	*pdata++ = (float)(d_histDist*d_euclidDist);
	*pdata++ = (float)(d_rectDist*d_euclidDist*d_histDist);

	return true;
}

/*
* 测试过程中获取光流距离
* 参数:
* 1.光流信息
* 2.前一帧图像Img_prev
* 3.前一帧中的检测框r_prev
* 4.后一帧中的检测框r_next
* 作用：
* 1.获取光流图像
* 2.根据光流计算Img_next中r_prev的中心位置
* 3.计算r_prevCenterPredict与r_nextCenter的欧式距离并高斯化
*/
bool CLogisticFeat::TestLogisticFeat(Mat flowInBox, Mat Img_prev, Rect r_prev, Rect r_next)
{
	//得到r_prev的中心点
	Point2f r_prevCenter((r_prev.tl() + r_prev.br()) / 2);
	Mat r_prevCenterMotionBoard = Mat::zeros(Img_prev.size() / GRID_SIZE, CV_32SC1);

	for (int y = 0; y < flowInBox.rows; y++)
	{
		for (int x = 0; x < flowInBox.cols; x++)
		{
			//各个像素的光流
			Point2f r_prevCenterMovingDir = flowInBox.at<Point2f>(y, x);
			//box中心点按光流运动到新位置
			Point2f r_prevNewCenter = r_prevCenter + r_prevCenterMovingDir;
																		 
			int moveRow = cvRound(r_prevNewCenter.y / GRID_SIZE);
			int moveCol = cvRound(r_prevNewCenter.x / GRID_SIZE);
			moveRow = max(moveRow, 0);
			moveCol = max(moveCol, 0);
			moveRow = min<int>(moveRow, r_prevCenterMotionBoard.rows - 1);
			moveCol = min<int>(moveCol, r_prevCenterMotionBoard.cols - 1);
			
			//预测位置记录加1
			++r_prevCenterMotionBoard.at<int>(moveRow, moveCol);
		}
	}

	Point r_prevCenterMostLikelyLoaction, useless;
	double useless1, useless2;
	minMaxLoc(r_prevCenterMotionBoard, &useless1, &useless2, &useless, &r_prevCenterMostLikelyLoaction);
	Point2f r_prevCenterPredict = r_prevCenterMostLikelyLoaction * GRID_SIZE;

	//得到原始box的运动预测位置
	r_prevCenterPredict.x += GRID_SIZE / 2;
	r_prevCenterPredict.y += GRID_SIZE / 2;

	Point2f r_nextCenter((r_next.tl() + r_next.br()) / 2);
	d_euclidDist = sqrt(pow(r_prevCenterPredict.x - r_nextCenter.x, 2) + pow(r_prevCenterPredict.y - r_nextCenter.y, 2));
	d_euclidDist = exp(-pow(d_euclidDist, 2) / pow(7, 2));

	flowInBox.release();
	r_prevCenterMotionBoard.release();

	return true;
}

/*
* 测试过程中获取特征信息
* 参数:
* 1.光流信息
* 2.前一帧图像Img_prev
* 3.前一帧中的检测框r_prev
* 4.后一帧中的检测框r_next
* 5.特征信息矩阵feature
* 作用：
* 1.获取光流距离特征
* 2.获取矩形框大小距离特征
* 3.获取颜色直方图距离特征
*/
bool CLogisticFeat::TestFeatInfo(Mat flowInBox, Mat Img_prev, Track prev, Track next, Mat &feature)
{
	if (!TestLogisticFeat(flowInBox, Img_prev, prev.box, next.box))
	{
		cout << "距离特征计算失败" << endl;
		return false;
	}

	feature.release();
	feature = Mat::zeros(1, 10, CV_32FC1);

	//矩形框大小相似度计算
	double d_rectDist = sqrt(pow(prev.box.x - next.box.x, 2) + pow(prev.box.y - next.box.y, 2)) / (prev.box.height / 2 + next.box.height / 2);
	//颜色直方图相似度计算
	d_histDist = compareHist(prev.hist, next.hist, CV_COMP_BHATTACHARYYA);

	float *pdata = feature.ptr<float>(0);
	*pdata++ = (float)d_rectDist;
	*pdata++ = (float)d_euclidDist;
	*pdata++ = (float)d_histDist;
	*pdata++ = (float)(d_rectDist*d_rectDist);
	*pdata++ = (float)(d_euclidDist*d_euclidDist);
	*pdata++ = (float)(d_histDist*d_histDist);
	*pdata++ = (float)(d_rectDist*d_euclidDist);
	*pdata++ = (float)(d_rectDist*d_histDist);
	*pdata++ = (float)(d_histDist*d_euclidDist);
	*pdata++ = (float)(d_rectDist*d_euclidDist*d_histDist);

	/*cout << feature.at<float>(0, 0) << endl;
	cout << feature.at<float>(0, 1) << endl;
	cout << feature.at<float>(0, 2) << endl;
	cout << feature.at<float>(0, 3) << endl;
	cout << feature.at<float>(0, 4) << endl;
	cout << feature.at<float>(0, 5) << endl;
	cout << feature.at<float>(0, 6) << endl;
	cout << feature.at<float>(0, 7) << endl;
	cout << feature.at<float>(0, 8) << endl;
	cout << feature.at<float>(0, 9) << endl;
	*/
	
	return true;
}

