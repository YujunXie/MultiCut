/*
* �������
* 1. �Ȱ�֡����С���������ٰ�id��С��������
* 2. �Ȱ�id��С���������ٰ�֡����С��������
*/

#include "FunctionDeclaration.h"

/*
* ������������Ȱ�֡����С���������ٰ�id��С��������
* ���ڽ������
*/
bool CmpResultToFile(const Track &a, const Track &b)
{
	if (a.frame != b.frame)
		return a.frame < b.frame;
	else
		return a.personId < b.personId;
}

/*
* ������������Ȱ�id��С���������ٰ�֡����С��������
* �������켣����
*/
bool CmpTrackletToFile(const Track &a, const Track &b)
{
	if (a.personId != b.personId)
		return a.personId < b.personId;
	else
		return a.frame < b.frame;
}

