/*
* 逻辑斯特分类模型的训练与测试头文件
*/

#include "DataType.h"
#include "LogisticFeat.h"
#include "LogisticReg.h"
#include "FunctionDeclaration.h"

bool trainLogistic();
void GenerateLRData(vector<Tracklet> trainTrackLet, Mat &trainData, Mat &trainLabel);