#include "opencv2\opencv.hpp"
#include "opencv2\nonfree\nonfree.hpp"

using namespace std;
using namespace cv;

class BoundaryCut
{
public:
	BoundaryCut();
	~BoundaryCut();

	void computeRightCut(int offset, Mat &overlapPatch, vector<vector<pair<int, int>>> &allMebc, Mat &srcImg, vector<pair<int, int>> &srcPch, int index);
	void computeDownCut(int offset, Mat &overlapPatch, vector<vector<pair<int, int>>> &allMebc, Mat &srcImg, vector<pair<int, int>> &srcPch, int index, int patchColNum);

private:

};

BoundaryCut::BoundaryCut()
{
}

BoundaryCut::~BoundaryCut()
{
}

void BoundaryCut::computeRightCut(int offset, Mat &overlapPatch, vector<vector<pair<int, int>>> &allMebc, Mat &srcImg, vector<pair<int, int>> &srcPch, int index)
{
	vector<vector<int>> rightCost(overlapPatch.rows, vector<int>(overlapPatch.cols, 0));
	vector<pair<int, int>> mebc(overlapPatch.rows, { 0, 0 });

	// distance map
	for (int ox = 0; ox < overlapPatch.rows; ox++)
	{
		for (int oy = 0; oy < overlapPatch.cols; oy++)
		{
			overlapPatch.at<uchar>(ox, oy) = sqrt(pow(srcImg.at<Vec3b>(srcPch[index - 1].first + ox, srcPch[index - 1].second + oy + offset)[0]
				- srcImg.at<Vec3b>(srcPch[index].first + ox, srcPch[index].second + oy)[0], 2)

				+ pow(srcImg.at<Vec3b>(srcPch[index - 1].first + ox, srcPch[index - 1].second + oy + offset)[1]
				- srcImg.at<Vec3b>(srcPch[index].first + ox, srcPch[index].second + oy)[1], 2)

				+ pow(srcImg.at<Vec3b>(srcPch[index - 1].first + ox, srcPch[index - 1].second + oy + offset)[2]
				- srcImg.at<Vec3b>(srcPch[index].first + ox, srcPch[index].second + oy)[2], 2)) / sqrt(255.0*255.0*3.0)*255.0;

		}
	}

	// dp
	for (int ox = 0; ox < overlapPatch.rows; ox++)
	{
		for (int oy = 0; oy < overlapPatch.cols; oy++)
		{
			if (ox == 0)
				rightCost[ox][oy] = (int)overlapPatch.at<uchar>(ox, oy);
			else
			{
				if (oy == 0)
				{
					rightCost[ox][oy] = (int)overlapPatch.at<uchar>(ox, oy)
						+ min(rightCost[ox - 1][oy], rightCost[ox - 1][oy + 1]);
				}
				else if (oy == overlapPatch.cols - 1)
				{
					rightCost[ox][oy] = (int)overlapPatch.at<uchar>(ox, oy)
						+ min(rightCost[ox - 1][oy - 1], rightCost[ox - 1][oy]);
				}
				else
				{
					rightCost[ox][oy] = (int)overlapPatch.at<uchar>(ox, oy)
						+ min(min(rightCost[ox - 1][oy - 1], rightCost[ox - 1][oy]), rightCost[ox - 1][oy + 1]);
				}
			}
		}
	}

	// trace back cut
	
	for (int ox = overlapPatch.rows - 1; ox >= 0; ox--)
	{
		int x, y;

		if (ox == overlapPatch.rows - 1)
		{
			int myMin = INT_MAX;
			for (int oy = 0; oy < overlapPatch.cols; oy++)
			{

				if (rightCost[ox][oy] < myMin)
				{
					myMin = rightCost[ox][oy];
					x = ox;
					y = oy;
				}
			}
		}
		else
		{
			int myMin = INT_MAX;

			for (int oy = -1; oy < 2; oy++)
			{
				int newOy = mebc[ox + 1].second + oy;
				if (newOy < 0 || newOy >= overlapPatch.cols)
					continue;

				if (rightCost[ox][newOy] < myMin)
				{
					myMin = rightCost[ox][newOy];
					x = ox;
					y = newOy;
				}
			}
		}
		mebc[ox] = { x, y };
	}

	allMebc.push_back(mebc);
}

void BoundaryCut::computeDownCut(int offset, Mat &overlapPatch, vector<vector<pair<int, int>>> &allMebc, Mat &srcImg, vector<pair<int, int>> &srcPch, int index, int patchColNum)
{
	vector<vector<int>> downCost(overlapPatch.rows, vector<int>(overlapPatch.cols, 0));
	int overlapPatchW = overlapPatch.rows;
	int patchN = overlapPatch.cols;
	vector<pair<int, int>> mebc(patchN, { 0, 0 });

	// distance map
	for (int ox = 0; ox < overlapPatchW; ox++)
	{
		for (int oy = 0; oy < patchN; oy++)
		{
			overlapPatch.at<uchar>(ox, oy) = sqrt(pow(srcImg.at<Vec3b>(srcPch[index].first + ox, srcPch[index].second + oy)[0]
				- srcImg.at<Vec3b>(srcPch[index - patchColNum].first + ox + offset, srcPch[index - patchColNum].second + oy)[0], 2)

				+ pow(srcImg.at<Vec3b>(srcPch[index].first + ox, srcPch[index].second + oy)[1]
				- srcImg.at<Vec3b>(srcPch[index - patchColNum].first + ox + offset, srcPch[index - patchColNum].second + oy)[1], 2)

				+ pow(srcImg.at<Vec3b>(srcPch[index].first + ox, srcPch[index].second + oy)[2]
				- srcImg.at<Vec3b>(srcPch[index - patchColNum].first + ox + offset, srcPch[index - patchColNum].second + oy)[2], 2)) / sqrt(255.0*255.0*3.0)*255.0;

		}
	}

	// dp
	for (int oy = 0; oy < patchN; oy++)
	{
		for (int ox = 0; ox < overlapPatchW; ox++)
		{
			if (oy == 0)
				downCost[ox][oy] = (int)overlapPatch.at<uchar>(ox, oy);
			else
			{
				if (ox == 0)
				{
					downCost[ox][oy] = (int)overlapPatch.at<uchar>(ox, oy)
						+ min(downCost[ox][oy - 1], downCost[ox + 1][oy - 1]);
				}
				else if (ox == overlapPatchW - 1)
				{
					downCost[ox][oy] = (int)overlapPatch.at<uchar>(ox, oy)
						+ min(downCost[ox - 1][oy - 1], downCost[ox][oy - 1]);
				}
				else
				{
					downCost[ox][oy] = (int)overlapPatch.at<uchar>(ox, oy)
						+ min(min(downCost[ox - 1][oy - 1], downCost[ox][oy - 1]), downCost[ox + 1][oy - 1]);
				}
			}
		}
	}

	// trace back cut
	
	for (int oy = patchN - 1; oy >= 0; oy--)
	{
		int x, y;

		if (oy == patchN - 1)
		{
			int myMin = INT_MAX;
			for (int ox = 0; ox < overlapPatchW; ox++)
			{

				if (downCost[ox][oy] < myMin)
				{
					myMin = downCost[ox][oy];
					x = ox;
					y = oy;
				}
			}
		}
		else
		{
			int myMin = INT_MAX;

			for (int ox = -1; ox < 2; ox++)
			{
				int newOx = mebc[oy + 1].first + ox;
				if (newOx < 0 || newOx >= overlapPatchW)
					continue;

				if (downCost[newOx][oy] < myMin)
				{
					myMin = downCost[newOx][oy];
					x = newOx;
					y = oy;
				}
			}
		}

		mebc[oy] = { x, y };
	}

	allMebc.push_back(mebc);
}