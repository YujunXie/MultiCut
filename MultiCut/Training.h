/*
* �߼�˹�ط���ģ�͵�ѵ�������ͷ�ļ�
*/

#include "DataType.h"
#include "LogisticFeat.h"
#include "LogisticReg.h"
#include "FunctionDeclaration.h"

bool trainLogistic();
void GenerateLRData(vector<Tracklet> trainTrackLet, Mat &trainData, Mat &trainLabel);