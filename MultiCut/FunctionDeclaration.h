/*
* �ļ���ȡ������Ԥ����ͷ�ļ�
*/

#include "DataType.h"

bool FileToTracklet(const string &filePath, vector<Tracklet>& TrackletSet);
bool FileToDetection(const string &filePath, vector<Track>& DetectionSet, int &nodesNum, bool flag);

bool TrackletToFile(vector<Track>& content, const string &filePath);

bool CmpResultToFile(const Track &a, const Track &b);
bool CmpTrackletToFile(const Track &a, const Track &b);