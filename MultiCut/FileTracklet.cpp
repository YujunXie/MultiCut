/*
* �ļ���ȡ������Ԥ����
*/
#ifndef FUNCTIONDECLARATION_H
#define FUNCTIONDECLARATION_H

#include "FunctionDeclaration.h"

/*
* ������ɫֱ��ͼ��Ϣ
* ����������temp
* ���ã�
* 1.�õ�������������ͼ��
* 2.��ͼ��ת��ΪHSV��ʽ
* 3.����Hͨ����Sͨ��bin����ȡֵ��Χ
* 4.������ɫֱ��ͼ��Ϣ����һ��
*/
void GetHistogram(Track& temp)
{
	char num1[100] = "";
	sprintf(num1, "%.6d", temp.frame);
	string frameFile = GROUND_TRUTH_IMAGE_FILE;
	Mat imageFrame = imread(frameFile + num1 + ".jpg");

	Rect rectNormal = temp.box;
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
	if ((rectNormal.y + rectNormal.height) > imageFrame.rows)
		rectNormal.height = imageFrame.rows - rectNormal.y;
	if ((rectNormal.x + rectNormal.width) > imageFrame.cols)
		rectNormal.width = imageFrame.cols - rectNormal.x;

	Mat src(imageFrame, Rect(rectNormal));
	Mat hsv;
	cvtColor(src, hsv, CV_BGR2HSV);
	
	// ��hueͨ��ʹ��30��bin,��saturatoinͨ��ʹ��32��bin
	int h_bins = 30; int s_bins = 32;
	int histSize[] = { h_bins, s_bins };

	// hue��ȡֵ��Χ��0��256, saturationȡֵ��Χ��0��180
	float h_ranges[] = { 0, 256 };
	float s_ranges[] = { 0, 180 };
	const float* ranges[] = { h_ranges, s_ranges };

	// ʹ�õ�0�͵�1ͨ��
	int channels[] = { 0, 1 };

	calcHist(&hsv, 1, channels, Mat(), temp.hist, 2, histSize, ranges, true, false);
	normalize(temp.hist, temp.hist, 0, 1, NORM_MINMAX, -1, Mat());
	
	//����hsv��ɫֱ��ͼ
	/*
	IplImage* src1 = cvLoadImage("E:/PAPER/Code/source/MOT16Labels/MOT16/train/MOT16-05/img1/000001.jpg");
	CvSize size = cvSize(12, 32);
	IplImage* roi = cvCreateImage(size, src1->depth, src1->nChannels);
	CvRect box = cvRect(307, 213, size.width, size.height);

	cvSetImageROI(src1, box);
	cvCopy(src1, roi);
	cvSaveImage("E:\\0.jpg", roi);
	cvResetImageROI(src1);
	cv::Mat src = cv::cvarrToMat(roi);
	Mat hsv;
	cvtColor(src, hsv, CV_BGR2HSV);

	int histSize[3];    
	float hranges[2];
	float sranges[2];
	float vranges[2];
	const float *ranges[3];
	int channels[3];
	int dims;

	histSize[0] = 30;
	histSize[1] = 32;
	histSize[2] = 32;
	hranges[0] = 0; hranges[1] = 256;
	sranges[0] = 0; sranges[1] = 180;
	vranges[0] = 0; vranges[1] = 256;
	ranges[0] = hranges;
	ranges[1] = sranges;
	ranges[2] = vranges;
	channels[0] = 0;
	channels[1] = 1;
	channels[3] = 2;
	dims = 3;

	Mat hist;
	calcHist(&hsv,1,channels,Mat(),hist,dims,histSize,ranges,true,false);

	int scale = 4;
	int hbins = histSize[0];
	int sbins = histSize[1];
	int vbins = histSize[2];
	float *hist_sta = new float[sbins];
	float *hist_val = new float[vbins];
	float *hist_hue = new float[hbins];
	memset(hist_val, 0, vbins * sizeof(float));
	memset(hist_sta, 0, sbins * sizeof(float));
	memset(hist_hue, 0, hbins * sizeof(float));

	for (int s = 0; s < sbins; s++)
	{
		for (int v = 0; v < vbins; v++)
		{
			for (int h = 0; h<hbins; h++)
			{
				float binVal = hist.at<float>(h, s);
				hist_hue[h] += binVal;
				//hist_val[v] += binVal;
				hist_sta[s] += binVal;
			}
		}
	}

	double max_sta = 0, max_hue = 0, max_val = 0;
	for (int i = 0; i<sbins; ++i)
	{
		if (hist_sta[i]>max_sta)
			max_sta = hist_sta[i];
	}
	for (int i = 0; i<hbins; ++i)
	{
		if (hist_hue[i]>max_hue)
			max_hue = hist_hue[i];
	}
	for (int i = 0; i<vbins; ++i)
	{
		if (hist_val[i]>max_val)
			max_val = hist_val[i];
	}

	Mat sta_img = Mat::zeros(240, sbins*scale + 20, CV_8UC3);
	Mat hue_img = Mat::zeros(240, hbins*scale + 20, CV_8UC3);
	Mat val_img = Mat::zeros(240, vbins*scale + 20, CV_8UC3);
	Mat img = Mat::zeros(240, hbins*scale + 20, CV_8UC3);

	for (int i = 0; i<sbins; ++i)
	{
		int intensity = cvRound(hist_sta[i] * (sta_img.rows - 10) / max_sta);
		rectangle(img, Point(i*scale + 10, sta_img.rows - intensity), Point((i + 1)*scale - 1 + 10, sta_img.rows - 1), Scalar(0, 0, 255), 1);
	}
	for (int i = 0; i<hbins; ++i)
	{
		int intensity = cvRound(hist_hue[i] * (hue_img.rows - 10) / max_hue);
		rectangle(img, Point(i*scale + 10, hue_img.rows - intensity), Point((i + 1)*scale - 1 + 10, hue_img.rows - 1), Scalar(255, 0, 0), 1);
	}
	for (int i = 0; i<vbins; ++i)
	{
		int intensity = cvRound(hist_val[i] * (val_img.rows - 10) / max_val);
		rectangle(img, Point(i*scale + 10, val_img.rows - intensity), Point((i + 1)*scale - 1 + 10, val_img.rows - 1), Scalar(0, 0, 255), 1);
	}

	/// ��ʾֱ��ͼ
	imshow("hist", img);
	waitKey(0);

	system("pause");
	*/
}

/*
* ���ַ����������������Ϣ
* ������
* 1.���룺�ַ���str
* 2.�������ʽ������dr
* 3.��ʶ���ж������ı���ʽflag. true: ground-truth�ı��ļ�; false: �����ı��ļ�;
*/
void Gentrack(const string &str, Track& dr, bool flag)
{
	char p[100] = { '\0' };
	int len = str.length();
	if (str.length() == 0)
	{
		dr.personId = -1;
		dr.frame = -1;
		return;
	}
	for (int i = 0, j = 0; i < len; ++i)
		if (str[i] != ' ')
		{
			if (!flag && j > 0 && p[j - 1] == ',')
			{
				p[j] = '-';
				p[++j] = '1';
				p[++j] = ',';
				++j;
				flag = true;
			}
			p[j] = str[i];
			++j;
		}
	sscanf_s(p, "%d,%d,%d,%d,%d,%d", &dr.frame, &dr.personId, &dr.box.x, &dr.box.y, &dr.box.width, &dr.box.height);
	
	/*int a, b, c;
	float x,y,w,h,pe;
	sscanf_s(p, "%d,%d,%f,%f,%f,%f,%f,%d,%d,%d", &dr.frame, &dr.personId, &x, &y, &w, &h, &pe, &a, &b, &c);
	if (pe < 0)
		dr.frame = -1;
	else {
		dr.box.x = cvFloor(x);
		dr.box.y = cvFloor(y);
		dr.box.width = cvFloor(w);
		dr.box.height = cvFloor(h);
	}
	
	/*int a, b;
	float c;
	sscanf_s(p, "%d,%d,%d,%d,%d,%d,%d,%d,%f", &dr.frame, &dr.personId, &dr.box.x, &dr.box.y, &dr.box.width, &dr.box.height, &a, &b, &c);
	if (a != 1 || b != 1 || c < 0.5)
		dr.personId = -1;
	*/	
}

/*
* ��ground-truth�ı��ļ������������������߼�˹�ط���ѵ��
* ������
* 1.���룺ground-truth�ı��ļ�·��filePath
* 2.������켣����TrackletSet
* ���ã�
* 1.�洢�ļ����ݣ�Ϊ��ȡtracklet��Ϣ��׼��
* 2.�ȶ�id����(����)���ٶ�frameIndex����(����)
* 3.��ȡtrack����Ϣ����tracklet
*/
bool FileToTracklet(const string &filePath, vector<Tracklet>& TrackletSet)
{
	ifstream file;
	file.open(filePath, file.in);
	if (!file.is_open())
	{
		cout << "FileToTracklet() open txt file failed" << endl;
		return false;
	}
	
	vector<Track> track;
	int lineNum = 0;
	while(lineNum <= 3000 && !file.eof())
	{
		string str;
		getline(file, str);
		Track temp;
		Gentrack(str, temp, true);
		if (temp.personId != -1)
		{
			cout << lineNum << endl;
			GetHistogram(temp);
			track.push_back(temp);
			lineNum++;
		}		
	}
	file.close();

	sort(track.begin(), track.end(), CmpTrackletToFile);
												
	Tracklet tempTracklet;
	for (auto det = track.begin(); det != track.end(); det++)
	{
		if (det == track.begin())
		{
			//track����֡
			if ((*det).personId == (*(det + 1)).personId)	
			{
				tempTracklet.firstFrame = (*det).frame;
				tempTracklet.trackInfo.push_back(*det);
			}
			//track������֡����β֡
			else	
			{
				tempTracklet.firstFrame = (*det).frame;
				tempTracklet.lastFrame = (*det).frame;
				tempTracklet.trackInfo.push_back(*det);
				tempTracklet.personId = (*det).personId;
				tempTracklet.isEnd = true;
				TrackletSet.push_back(tempTracklet);
				tempTracklet.trackInfo.clear();
			}
		}
		else if (det + 1 == track.end())
		{
			//track��β֡
			if ((*(det - 1)).personId == (*det).personId)	
			{
				tempTracklet.lastFrame = (*det).frame;
				tempTracklet.trackInfo.push_back(*det);
				tempTracklet.personId = (*det).personId;
				tempTracklet.isEnd = true;
				TrackletSet.push_back(tempTracklet);
				tempTracklet.trackInfo.clear();
			}
			//track������֡����β֡
			else	
			{
				tempTracklet.firstFrame = (*det).frame;
				tempTracklet.lastFrame = (*det).frame;
				tempTracklet.trackInfo.push_back(*det);
				tempTracklet.personId = (*det).personId;
				tempTracklet.isEnd = true;
				TrackletSet.push_back(tempTracklet);
				tempTracklet.trackInfo.clear();
			}
		}
		else
		{
			//track����֡
			if ((*(det - 1)).personId != (*det).personId && (*det).personId == (*(det + 1)).personId)	
			{
				tempTracklet.firstFrame = (*det).frame;
				tempTracklet.trackInfo.push_back(*det);
			}
			//track���м�֡
			if ((*(det - 1)).personId == (*det).personId && (*det).personId == (*(det + 1)).personId)	
			{
				tempTracklet.trackInfo.push_back(*det);
			}
			//track��β֡
			if ((*(det - 1)).personId == (*det).personId && (*det).personId != (*(det + 1)).personId)	
			{
				tempTracklet.lastFrame = (*det).frame;
				tempTracklet.trackInfo.push_back(*det);
				tempTracklet.personId = (*det).personId;
				tempTracklet.isEnd = true;
				TrackletSet.push_back(tempTracklet);
				tempTracklet.trackInfo.clear();
			}
			//track������֡����β֡
			if ((*(det - 1)).personId != (*det).personId && (*det).personId != (*(det + 1)).personId)	
			{
				tempTracklet.firstFrame = (*det).frame;
				tempTracklet.lastFrame = (*det).frame;
				tempTracklet.trackInfo.push_back(*det);
				tempTracklet.personId = (*det).personId;
				tempTracklet.isEnd = true;
				TrackletSet.push_back(tempTracklet);
				tempTracklet.trackInfo.clear();
			}
		}
		cout << (*det).frame << " " << (*det).personId << " " << (*det).box << endl;
	}
	return true;
}

/*
* �ɼ����ı��ļ��õ����м���
* ������
* 1.���룺�����ı��ļ�·��filePath
* 2.��������򼯺�detectionSet
* 3.�������nodesNum
* 4.��ʶ��flag. true�����ӻ�����ļ���false�������ı��ļ���
* ���ã��洢�ļ�����,Ϊ��ȡdetection��Ϣ��׼��
*/
bool FileToDetection(const string &filePath, vector<Track>& detectionSet, int &nodesNum, bool flag)
{
	ifstream file;
	file.open(filePath, file.in);
	if (!file.is_open())
	{
		cout << "FileToDetection() open txt file failed" << endl;
		return false;
	}

	/*�洢�ļ�����Ϊ��ȡdetection��Ϣ��׼��*/
	while (!file.eof())
	{
		string str;
		getline(file, str);
		Track temp;
		Gentrack(str, temp, flag);
		if (temp.frame != -1)
		{
			cout << nodesNum << endl;
			//GetHistogram(temp);
			detectionSet.push_back(temp);
			nodesNum++;
		}
	}

	file.close();
	return true;
}

/*
* �����и������ݵõ��켣����ı��ļ������ڷ�����������ӻ�
* ������
* 1.���룺�켣����trackletSet
* 2.������켣����ı��ļ�·��filePath
*/
bool TrackletToFile(vector<Track>& content, const string &filePath)
{
	/*vector<Track> content;
	for (auto i = trackletSet.begin(); i != trackletSet.end(); i++)
	{
		for (; !(*i).trackInfo.empty();)
		{
			content.push_back((*i).trackInfo.back());	//����tracklet�е�track,ȫ��ת����
			(*i).trackInfo.pop_back();
		}
	}*/
	
	ofstream file;
	file.open(filePath, ios::trunc);
	if (!file.is_open())
		return false;

	for (auto i = content.begin(); i != content.end(); i++)
	{
		if ((i + 1) != content.end() && (*i).frame == (*(i+1)).frame && (*i).personId == (*(i + 1)).personId)
		{ 
			file << (*i).frame << "," << (*i).personId << "," << ((*i).box.x + (*(i + 1)).box.x)/2 << "," << ((*i).box.y + (*(i + 1)).box.y) / 2 << "," << ((*i).box.width + (*(i + 1)).box.width) / 2 << "," << ((*i).box.height + (*(i + 1)).box.height) / 2 << endl;
			while ((i + 1) != content.end() &&(*(i + 1)).personId == (*i).personId)
				i++;
		}
		
		else file << (*i).frame << "," << (*i).personId << "," << (*i).box.x << "," << (*i).box.y << "," << (*i).box.width << "," << (*i).box.height << endl;
	}
	
	if (file)
		file.close();
	return true;
}



#endif