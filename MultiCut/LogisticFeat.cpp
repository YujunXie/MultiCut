/*
* ����ģ�;������
* 1.����A�������С���ƶ�LogisticFeat
* 2.����B�����ģ��
* 3.����C���˶�ģ��
*/

#include "LogisticFeat.h"

CLogisticFeat::CLogisticFeat()
{

}

CLogisticFeat::~CLogisticFeat()
{

}

/*
* ��ȡ����ͼ��
* ����:
* 1.ǰһ֡ͼ��Img_prev
* 2.��һ֡ͼ��Img_next
* ���ã�
* 1.����ɫͼת���ɻҶ�ͼ
* 2.������ܹ���
* 3.����ɹ�ֱ���ͷ�ԭ�ȵĹ�������
*/
bool CLogisticFeat::GetOpticalFlowMap(Mat Img_prev, Mat Img_next)
{
	if (Img_prev.empty() || Img_next.empty())
		cout << "�����˿�ͼ��" << endl;
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
		cout << "�������ʧ��" << endl;
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
* ѵ�������л�ȡ��������
* ����:
* 1.ǰһ֡ͼ��Img_prev
* 2.��һ֡ͼ��Img_next
* 3.ǰһ֡�еļ���r_prev
* 4.��һ֡�еļ���r_next
* ���ã�
* 1.��ȡ����ͼ��
* 2.���ݹ�������Img_next��r_prev������λ��
* 3.����r_prevCenterPredict��r_nextCenter��ŷʽ���벢��˹��
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


	//�õ�r_prev�����ĵ�
	Point2f r_prevCenter((rectNormal.tl() + rectNormal.br()) / 2);
	//����boundingbox�еĹ���
	Mat flowInBox(m_opticalFlow(rectNormal));
	Mat r_prevCenterMotionBoard = Mat::zeros(Img_prev.size() / GRID_SIZE, CV_32SC1);

	for (int y = 0; y < flowInBox.rows; y++)
	{
		for (int x = 0; x < flowInBox.cols; x++)
		{
			//�������صĹ���
			Point2f r_prevCenterMovingDir = flowInBox.at<Point2f>(y, x);
			//box���ĵ㰴�����˶�����λ��
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

			//Ԥ��λ�ü�¼��1
			++r_prevCenterMotionBoard.at<int>(moveRow, moveCol);
		}
	}

	Point r_prevCenterMostLikelyLoaction, useless;
	double useless1, useless2;
	minMaxLoc(r_prevCenterMotionBoard, &useless1, &useless2, &useless, &r_prevCenterMostLikelyLoaction);
	Point2f r_prevCenterPredict = r_prevCenterMostLikelyLoaction * GRID_SIZE;

	//�õ�ԭʼbox���˶�Ԥ��λ��
	r_prevCenterPredict.x += GRID_SIZE / 2;
	r_prevCenterPredict.y += GRID_SIZE / 2;

	Point2f r_nextCenter((r_next.tl() + r_next.br()) / 2);
	d_euclidDist = sqrt(pow(r_prevCenterPredict.x - r_nextCenter.x, 2) + pow(r_prevCenterPredict.y - r_nextCenter.y, 2));
	//ŷʽ�����˹��
	d_euclidDist = exp(-pow(d_euclidDist, 2) / pow(7, 2));

	flowInBox.release();
	r_prevCenterMotionBoard.release();

	return true;
}

/*
* ѵ�������л�ȡ������Ϣ
* ����:
* 1.ǰһ֡ͼ��Img_prev
* 2.��һ֡ͼ��Img_next
* 3.ǰһ֡�еļ���r_prev
* 4.��һ֡�еļ���r_next
* 5.������Ϣ����feature
* ���ã�
* 1.��ȡ������������
* 2.��ȡ���ο��С��������
* 3.��ȡ��ɫֱ��ͼ��������
*/
bool CLogisticFeat::TrainFeatInfo(Mat Img_prev, Mat Img_next, Track prev, Track next, Mat &feature)
{
	if (!TrainLogisticFeat(Img_prev, Img_next, prev.box, next.box))
	{
		cout << "������������ʧ��" << endl;
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
* ���Թ����л�ȡ��������
* ����:
* 1.������Ϣ
* 2.ǰһ֡ͼ��Img_prev
* 3.ǰһ֡�еļ���r_prev
* 4.��һ֡�еļ���r_next
* ���ã�
* 1.��ȡ����ͼ��
* 2.���ݹ�������Img_next��r_prev������λ��
* 3.����r_prevCenterPredict��r_nextCenter��ŷʽ���벢��˹��
*/
bool CLogisticFeat::TestLogisticFeat(Mat flowInBox, Mat Img_prev, Rect r_prev, Rect r_next)
{
	//�õ�r_prev�����ĵ�
	Point2f r_prevCenter((r_prev.tl() + r_prev.br()) / 2);
	Mat r_prevCenterMotionBoard = Mat::zeros(Img_prev.size() / GRID_SIZE, CV_32SC1);

	for (int y = 0; y < flowInBox.rows; y++)
	{
		for (int x = 0; x < flowInBox.cols; x++)
		{
			//�������صĹ���
			Point2f r_prevCenterMovingDir = flowInBox.at<Point2f>(y, x);
			//box���ĵ㰴�����˶�����λ��
			Point2f r_prevNewCenter = r_prevCenter + r_prevCenterMovingDir;
																		 
			int moveRow = cvRound(r_prevNewCenter.y / GRID_SIZE);
			int moveCol = cvRound(r_prevNewCenter.x / GRID_SIZE);
			moveRow = max(moveRow, 0);
			moveCol = max(moveCol, 0);
			moveRow = min<int>(moveRow, r_prevCenterMotionBoard.rows - 1);
			moveCol = min<int>(moveCol, r_prevCenterMotionBoard.cols - 1);
			
			//Ԥ��λ�ü�¼��1
			++r_prevCenterMotionBoard.at<int>(moveRow, moveCol);
		}
	}

	Point r_prevCenterMostLikelyLoaction, useless;
	double useless1, useless2;
	minMaxLoc(r_prevCenterMotionBoard, &useless1, &useless2, &useless, &r_prevCenterMostLikelyLoaction);
	Point2f r_prevCenterPredict = r_prevCenterMostLikelyLoaction * GRID_SIZE;

	//�õ�ԭʼbox���˶�Ԥ��λ��
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
* ���Թ����л�ȡ������Ϣ
* ����:
* 1.������Ϣ
* 2.ǰһ֡ͼ��Img_prev
* 3.ǰһ֡�еļ���r_prev
* 4.��һ֡�еļ���r_next
* 5.������Ϣ����feature
* ���ã�
* 1.��ȡ������������
* 2.��ȡ���ο��С��������
* 3.��ȡ��ɫֱ��ͼ��������
*/
bool CLogisticFeat::TestFeatInfo(Mat flowInBox, Mat Img_prev, Track prev, Track next, Mat &feature)
{
	if (!TestLogisticFeat(flowInBox, Img_prev, prev.box, next.box))
	{
		cout << "������������ʧ��" << endl;
		return false;
	}

	feature.release();
	feature = Mat::zeros(1, 10, CV_32FC1);

	//���ο��С���ƶȼ���
	double d_rectDist = sqrt(pow(prev.box.x - next.box.x, 2) + pow(prev.box.y - next.box.y, 2)) / (prev.box.height / 2 + next.box.height / 2);
	//��ɫֱ��ͼ���ƶȼ���
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

