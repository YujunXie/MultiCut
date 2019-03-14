/*
* ������
*/

#include "DataType.h"
#include "LogisticFeat.h"
#include "LogisticReg.h"
#include "Training.h"
#include "FunctionDeclaration.h"

/*
* �ɼ����ı��ļ��õ����м���
* ������
* 1.���룺���򼯺�detectionSet
* 2.������������nodesNum
*/
bool getDetections(vector<Track>& detectionSet, int &nodesNum)
{
	if (!FileToDetection(DETECTION_FILE, detectionSet, nodesNum, true))
		return false;

	cout << nodesNum << endl;

}

/*
* �����������
* ������
* 1.��ʼ֡��begin
* 2.ÿ֡�м������framesNum[]
* 3.����sub
* 4.���򼯺�detectionSet
* 5.��������opticalFlow
* ����
* 1.����ÿһ֡��ͼ��ʱ��������֡�������ͼ��
* 2.���������ͼ���Ĺ�����Ϣ���洢�ڶ�Ӧ����µ�������
* 3.�������������±���ж�ȡ����ֵ
*/
bool calOpticalFlow(int begin, int framesNum[], int sub, int nodesNum, vector<Track> &detectionSet, vector< vector<Mat> > &opticalFlow)
{
	vector<Track>::iterator iter_i = detectionSet.begin();
	vector<Track>::iterator iter_j;
	CLogisticFeat *logisFeat = new CLogisticFeat;
	string frameFile = GROUND_TRUTH_IMAGE_FILE;

	for (int i = 0; i < FRAMENUM - 1; i++)
	{
		iter_i += begin;
		iter_j = iter_i;
		char num1[100] = "";
		sprintf(num1, "%.6d", (*iter_i).frame);
		Mat imagePrev = imread(frameFile + num1 + ".jpg");

		//cout << begin << " " << (*iter_i).frame << endl;

		for (int j = i; j < FRAMENUM - 1; j++)
		{
			iter_j += framesNum[j + 1];
			
			char num2[100] = ""; 
			sprintf(num2, "%.6d", (*(iter_j)).frame);
			Mat imageNext = imread(frameFile + num2 + ".jpg");

			logisFeat->GetOpticalFlowMap(imagePrev, imageNext);
			/*int x = (*iter_i).frame % sub;
			int y = (*iter_j).frame % sub;
			opticalFlow[x * y] = logisFeat->m_opticalFlow;*/
			
			opticalFlow[(*iter_i).frame][(*iter_j).frame] = logisFeat->m_opticalFlow;
			
			//cout << (*iter_j).frame << " " << x*y << endl;
		}
		begin = framesNum[i + 1];
	}

	return true;
}

/*
* ÿ5֡����һ��ͼ�ṹ
* ������
* 1.��ʼ֡��begin
* 2.ÿ֡�м������framesNum[]
* 3.��ͼ�����м�����������ڵ������
* 4.���򼯺�detectionSet
* ����
* 1.����һ��ͼ�Լ����бߵ�Ȩֵ����
* 2.�����������Ϣ����
* 3.����ÿһ֡�ļ���ʱ��������֡������м���
* 3.�������������������ı�,�����������ϵ�Ȩֵ
* 4.ʵ��ͼ���㷨������ID����
*/
bool BuildGraph(int begin, int framesNum[], int nodesNum, vector<Track> &detectionSet, int &max_id)
{
	andres::graph::Graph<> graph;
	graph.insertVertices(nodesNum*100 + begin);
	int edgesNum = 0;
	for (int m = 1; m <= FRAMENUM - 1; m++)
		for (int n = m + 1; n <= FRAMENUM; n++)
			edgesNum += framesNum[m] * framesNum[n];

	//andres::graph::Graph<> original_graph(nodesNum * 100 + begin);
	//andres::graph::CompleteGraph<> lifted_graph(original_graph.numberOfVertices());
	
	vector<double> weights(edgesNum);
	int i = begin, j = 0, k = 0, t = 0, nums = begin;
	int x[90000], y[90000];
	vector<Track>::iterator iter_i = detectionSet.begin() + begin;
	vector<Track>::iterator iter_j;
	vector<Track>::iterator iter_num = detectionSet.begin() + begin + nodesNum;
	CLogisticFeat *logisFeat = new CLogisticFeat;
	CLogisticReg *logisReg = new CLogisticReg;
	//vector<Mat> opticalFlow(1000);
	vector< vector<Mat> > opticalFlow(2000);
	for (int m = 0; m < 2000; m++)
		opticalFlow[m].resize(2000);

	int sub = (*iter_i).frame - 1;
	if (sub < 5)
		sub = 10;
	
	calOpticalFlow(begin, framesNum, sub, nodesNum, detectionSet, opticalFlow);
	
	for (; iter_i != iter_num - 1; iter_i++)
	{
		if (i >= nums)
		{
			t++;
			j = i + framesNum[t];
			iter_j = iter_i + framesNum[t];
			nums += framesNum[t];
		}
		else
		{
			j = nums;
			iter_j = detectionSet.begin() + nums;
		}

		char num1[100] = "";
		sprintf(num1, "%.6d", (*iter_i).frame);
		string frameFile = GROUND_TRUTH_IMAGE_FILE;
		Mat imagePrev = imread(frameFile + num1 + ".jpg");
	
		for (; iter_j != iter_num; iter_j++)
		{
			graph.insertEdge(i, j);
			//original_graph.insertEdge(i, j);
			
			Rect rectPrev, rectNext;
			rectPrev = (*iter_i).box;
			rectNext = (*iter_j).box;
			Rect rectNormal = rectPrev;

			if (rectPrev.x < 0)
			{
				rectNormal.width -= rectNormal.x;
				rectNormal.x = 0;
			}
			if (rectPrev.y < 0)
			{
				rectNormal.height -= rectNormal.y;
				rectNormal.y = 0;
			}
			if ((rectNormal.y + rectNormal.height) > imagePrev.rows)
				rectNormal.height = imagePrev.rows - rectNormal.y;
			if ((rectNormal.x + rectNormal.width) > imagePrev.cols)
				rectNormal.width = imagePrev.cols - rectNormal.x;

			Mat featMat;
			Mat response;
			float pos;

			//int a = (*iter_i).frame % sub;
			//int b = (*iter_j).frame % sub;
			//Mat flowInBox(opticalFlow[a * b](rectNormal));
			
			Mat flowInBox(opticalFlow[(*iter_i).frame][(*iter_j).frame](rectNormal));
			logisFeat->TestFeatInfo(flowInBox, imagePrev, *iter_i, *iter_j, featMat);
			logisReg->LogisticPredict(featMat, response, pos);

			double weight = log(pos / (1 - pos));
			x[k] = i;
			y[k] = j;

			cout << k << " " << i << " " << j << " " << weight << "  " << response << "  " << (*iter_i).box << "  " << (*iter_j).box << endl;

			weights[k++] = weight;

			//weights[lifted_graph.findEdge(i, j).second] = weight;
			j++;
			
		}
		i++;
	}

	vector<char> edge_labels(graph.numberOfEdges());
	andres::graph::multicut::kernighanLin(graph, weights, edge_labels, edge_labels);
	//cout << lifted_graph.numberOfEdges() << endl;
	//vector<char> edge_labels(lifted_graph.numberOfEdges(),1);
	//andres::graph::multicut_lifted::kernighanLin(original_graph, lifted_graph, weights, edge_labels, edge_labels);
	//cout << "yes2" << endl;
	int id = max_id;
	for(k = 0; k < edgesNum; k++)
	{
		if(detectionSet[x[k]].personId == -1)
			detectionSet[x[k]].personId = ++id;
		if(edge_labels[k] == 0)
		//if (edge_labels[lifted_graph.findEdge(x[k],y[k]).second] == 0)
		{
			detectionSet[y[k]].personId = detectionSet[x[k]].personId;
		}
		id = max(id, detectionSet[x[k]].personId);
	}
	max_id = id;
	/*for (i = begin; i < begin + nodesNum; i++)
	{
		cout << detectionSet[i].frame << " " << detectionSet[i].personId << " " << detectionSet[i].box << endl;
	}
	system("pause");*/
	return true;
}

/*
* ʵ����ͼ�����㷨
* ������
* 1.���򼯺�detectionSet
* 2.���м������nodesNum
* ����
* 1.��frameIndex�������м���
* 2.ÿ5֡����һ��ͼ�ṹ
* 3.ǰһ��ͼ��β֡��Ϊ��һ��ͼ����֡������һ��ͼ��ֱ�����м��򱻸���ID
*/
bool MultiCut(vector<Track>& detectionSet, int &nodesNum)
{
	nodesNum = 0;
	int max_id = 0;
	int framesNum[FRAMENUM + 1] = { 0 };

	int begin = 0, frame_begin = (*detectionSet.begin()).frame, frame_end = 0, lastFrameNum = 0;
	int cnt = 0;
	vector<Track>::iterator iter_j = detectionSet.begin();
	frame_begin = (*iter_j).frame;

	for (; iter_j != detectionSet.end(); iter_j++)
	{
		frame_end = (*iter_j).frame;
		int nums = frame_end - frame_begin;

		if (nums >= 1 && cnt < FRAMENUM + 1 && framesNum[cnt + 1] == 0)
		{
			
			cnt++;
			if (cnt < FRAMENUM)
			{
				int t = 0;
				for (int i = cnt - 1; i >= 0; i--)
					t += framesNum[i];
				framesNum[cnt] = nodesNum - t;
				frame_begin = frame_end;
			}
			else {
				framesNum[cnt] = lastFrameNum + 1;
				//cout << begin << " " << frame_end << " " << frame_begin << " " << framesNum[1] << " " << framesNum[2] << " " << framesNum[3] << " " << framesNum[4] << " " << framesNum[5] << " " << nodesNum << endl;

				BuildGraph(begin, framesNum, nodesNum, detectionSet, max_id);

				for (int m = FRAMENUM; m > 4; m--)
					nodesNum -= framesNum[m];
				begin += nodesNum;

				//begin += nodesNum - framesNum[FRAMENUM];
				memset(framesNum, 0, sizeof(framesNum));
				nodesNum = 0;
				cnt = 0;
				iter_j = detectionSet.begin() + begin;
				frame_begin = (*iter_j).frame;
				lastFrameNum = 0;
				//cout << begin << " " << framesNum[1] << " " << frame_begin << " " << (*(iter_j + 1)).frame << nodesNum << endl;
			}

		}
		
		/*if (framesNum[nums] == 0 && nums < FRAMENUM + 1 && nums > 0)
		{
			if (nums < FRAMENUM)
			{
				int t = 0;
				for (int i = nums - 1; i >= 0; i--)
					t += framesNum[i];
				framesNum[nums] = nodesNum - t;
			}
			else {
				framesNum[nums] = lastFrameNum + 1;
				cout << begin << " " << frame_end << " " << frame_begin << " " << framesNum[1] << " " << framesNum[2] << " " << framesNum[3] << " " << framesNum[4] << " " << framesNum[5] << " " << nodesNum << endl;

				BuildGraph(begin, framesNum, nodesNum, detectionSet, max_id);

				begin += nodesNum - framesNum[FRAMENUM];
				memset(framesNum, 0, sizeof(framesNum));
				nodesNum = 0;
				frame_begin = frame_end - 1;
				iter_j = detectionSet.begin() + begin;
				lastFrameNum = 0;
				//cout << begin << " " << framesNum[1] << " " << frame_begin << " " << (*iter_j).frame << " " << nodesNum << endl;
				//system("pause");
			}
		}
		*/
		else if (cnt == FRAMENUM - 1)
			lastFrameNum++;

		nodesNum++;
	}
	return true;
}

/*
* �������й켣
* ���������򼯺�detectionSet
* ���ã�
* 1.�ȶ�id����(����)���ٶ�frameIndex����(����)
* 2.��ȡtrack����Ϣ����tracklet����ʱ���ã�
* 3.�洢����track��Ϣ
*/
bool Trajectory(vector<Track> &detectionSet)
{
	sort(detectionSet.begin(), detectionSet.end(), CmpResultToFile);
	
	vector<Tracklet>TrackletSet;
	Tracklet tempTracklet;
	for (auto det = detectionSet.begin(); det != detectionSet.end(); det++)
	{
		if (det == detectionSet.begin())
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
		else if (det + 1 == detectionSet.end())
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
	
	TrackletToFile(detectionSet, VISIUAL_RESULT_FILE);
	return true;
}

/*
* ��id�ľ��ο����
* ����:
* 1.֡frame
* 2.��֡������Ŀ����Ϣtrack
* 3.��ɫ����colors
* ���ã�
* 1.������������ӣ����������ɫ
* 2.��ͬidĿ����ɫ������ͬ��������£���ͬidĿ�������������ɫ
* 3.���ƾ��ο������
*/
bool MoveDetect(Mat &frame, vector<Track> track, Scalar colors[])
{
	RNG rng(time(0));
	vector<Track>::iterator iter_i;
	for (iter_i = track.begin(); iter_i != track.end(); iter_i++)
	{
		int id = (*iter_i).personId;
		if (colors[id] == Scalar(0, 0, 0))
			colors[id] = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Rect box = (*iter_i).box;
		int baseline;
		stringstream ss;
		ss << id;
		string text = "id=" + ss.str();
		rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), colors[id], 2, 8);
		Size text_size = getTextSize(text, CV_FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
		//putText(frame, text, Point(box.x + box.width / 2 - text_size.width / 2, box.y - text_size.height / 2), CV_FONT_HERSHEY_SIMPLEX, 0.8f, colors[id], 2);
		putText(frame, text, Point(box.x + box.width / 2, box.y - text_size.height / 2), CV_FONT_HERSHEY_DUPLEX, 0.7f, colors[id], 2);
	}
	return true;
}

/*
* ���ӻ�ÿ֡������Ŀ��
* ���ã�
* 1.�ӿ��ӻ�����ļ��ж�ȡ���м�����Ϣ
* 2.��֡��ȡͼ���������Ŀ��Ļ���
*/
bool Show()
{
	Mat frame;
	Mat result;
	string frameFile = GROUND_TRUTH_IMAGE_FILE;

	int nums = 0;
	vector<Track> Tracks;
	vector<Track> content;
	FileToDetection(VISIUAL_RESULT_FILE, Tracks, nums, true);

	auto i = Tracks.begin();
	int prev_frame = (*i).frame;
	char num1[100] = "";
	sprintf(num1, "%.6d", prev_frame);
	Mat img = imread(frameFile + num1 + ".jpg");
	Scalar colors[1000] = { Scalar(0,0,0) };

	for (; i != Tracks.end(); i++)
	{
		cout << (*i).frame << " " << (*i).personId << " " << (*i).box << endl;
		content.push_back((*i));
		if ((*i).frame != prev_frame)
		{
			MoveDetect(img, content, colors);
			namedWindow("result", 0);
			resizeWindow("result", 980, 580);
			imshow("result", img);
			waitKey(1000);
			
			prev_frame = (*i).frame;
			sprintf(num1, "%.6d", prev_frame);
			img = imread(frameFile + num1 + ".jpg");
			content.clear();
			content.push_back((*i));
		}
	}

	return true;
}

int main(int argc, char* argv)
{
	/* 1. training */
	//trainLogistic();

	/* 2. get detection datas */
	int nodesNum = 0;
	vector<Track> detectionSet;
	//getDetections(detectionSet, nodesNum);

	/* 3. multi-cut algorithmn */
	//MultiCut(detectionSet, nodesNum);
	
	/* 4. obtain trajectories */
	//Trajectory(detectionSet);

	/* 5. show results */
	Show();

	system("pause");
		
	/* ����������� */

	/*ifstream file;
	file.open("E:/PAPER/Code/source/MOT16Labels/train/MOT16-05/gt/gt.txt", file.in);
	if (!file.is_open())
	{
		cout << "FileToDetection() open txt file failed" << endl;
		return false;
	}
	vector<Track> track;
	while (!file.eof())
	{
		string str;
		getline(file, str);
		Track temp;
		int a, b;
		float c;
		sscanf_s(str.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%f", &temp.frame, &temp.personId, &temp.box.x, &temp.box.y, &temp.box.width, &temp.box.height, &a, &b, &c);
		if (a != 1 || b != 1 || c < 0.5)
			continue;
		track.push_back(temp);
	}
	file.close();

	sort(track.begin(), track.end(), CmpResultToFile);

	vector<Track>::iterator iter;
	ofstream file1;
	ofstream file2;
	file1.open("E:/PAPER/Code/Test/MultiCut/Data/MOT16-05/dt.txt", ios::trunc);
	if (!file1.is_open())
		return false;
	file2.open("E:/PAPER/Code/Test/MultiCut/Data/MOT16-05/gt_dt.txt", ios::trunc);
	if (!file2.is_open())
		return false;

	for (iter = track.begin(); iter != track.end(); iter++)
	{
		cout << (*iter).frame << "," << (*iter).box.x << "," << (*iter).box.y << "," << (*iter).box.width << "," << (*iter).box.height << endl;
		file1 << (*iter).frame << "," << (*iter).box.x << "," << (*iter).box.y << "," << (*iter).box.width << "," << (*iter).box.height << endl;
		file2 << (*iter).frame << "," << (*iter).personId << "," << (*iter).box.x << "," << (*iter).box.y << "," << (*iter).box.width << "," << (*iter).box.height << endl;
	}
	if (file1)
		file1.close();
	if (file2)
		file1.close();

	system("pause");
	*/
	return 0;
}