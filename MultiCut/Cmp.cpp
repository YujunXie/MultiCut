/*
* 排序规则：
* 1. 先按帧数从小到大排序，再按id从小到大排序
* 2. 先按id从小到大排序，再按帧数从小到大排序
*/

#include "FunctionDeclaration.h"

/*
* 检测框排序规则：先按帧数从小到大排序，再按id从小到大排序
* 用于结果评测
*/
bool CmpResultToFile(const Track &a, const Track &b)
{
	if (a.frame != b.frame)
		return a.frame < b.frame;
	else
		return a.personId < b.personId;
}

/*
* 检测框排序规则：先按id从小到大排序，再按帧数从小到大排序
* 用于最后轨迹生成
*/
bool CmpTrackletToFile(const Track &a, const Track &b)
{
	if (a.personId != b.personId)
		return a.personId < b.personId;
	else
		return a.frame < b.frame;
}

