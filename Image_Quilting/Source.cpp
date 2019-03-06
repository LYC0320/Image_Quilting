#include <iostream>
#include "opencv2\opencv.hpp"
#include "opencv2\nonfree\nonfree.hpp"

#include <math.h>
#include <time.h>
#include <vector>

using namespace std;
using namespace cv;

pair<int, int> SSD(Mat s, Mat t, int patchN)
{
	float myMin = INT_MAX;
	pair<int, int> p;

	for (int ox = 0; ox <= s.rows - patchN; ox++)
	{
		for (int oy = 0; oy <= s.cols - patchN; oy++)
		{
			float d = 0;

			for (int ix = 0; ix < t.rows; ix++)
			{
				for (int iy = 0; iy < t.cols; iy++)
				{
					d += sqrt(pow((float)s.at<Vec3b>(ox + ix, oy + iy)[0] - (float)t.at<Vec3b>(ix, iy)[0], 2)
						+ pow((float)s.at<Vec3b>(ox + ix, oy + iy)[1] - (float)t.at<Vec3b>(ix, iy)[1], 2)
						+ pow((float)s.at<Vec3b>(ox + ix, oy + iy)[2] - (float)t.at<Vec3b>(ix, iy)[2], 2));
				}
			}

			if (d < myMin)
			{
				myMin = d;
				p = { ox, oy };
			}
		}
	}
	cout << "Single:" << p.first << "," << p.second << ":" << myMin << endl;

	return p;
}

pair<int, int> BSSD(Mat s, Mat t1, Mat t2, int patchN)
{
	float myMin = INT_MAX;
	pair<int, int> p;

	for (int ox = 0; ox <= s.rows - patchN; ox++)
	{
		for (int oy = 0; oy <= s.cols - patchN; oy++)
		{
			float d = 0;

			for (int ix = 0; ix < t1.rows; ix++)
			{
				for (int iy = 0; iy < t1.cols; iy++)
				{
					d += sqrt(pow((float)s.at<Vec3b>(ox + ix, oy + iy)[0] - (float)t1.at<Vec3b>(ix, iy)[0], 2)
						+ pow((float)s.at<Vec3b>(ox + ix, oy + iy)[1] - (float)t1.at<Vec3b>(ix, iy)[1], 2)
						+ pow((float)s.at<Vec3b>(ox + ix, oy + iy)[2] - (float)t1.at<Vec3b>(ix, iy)[2], 2));
				}
			}

			for (int ix = 0; ix < t2.rows; ix++)
			{
				for (int iy = 0; iy < t2.cols; iy++)
				{
					d += sqrt(pow((float)s.at<Vec3b>(ox + ix, oy + iy)[0] - (float)t2.at<Vec3b>(ix, iy)[0], 2)
						+ pow((float)s.at<Vec3b>(ox + ix, oy + iy)[1] - (float)t2.at<Vec3b>(ix, iy)[1], 2)
						+ pow((float)s.at<Vec3b>(ox + ix, oy + iy)[2] - (float)t2.at<Vec3b>(ix, iy)[2], 2));
				}
			}

			if (d < myMin)
			{
				myMin = d;
				p = { ox, oy };
			}
		}
	}
	cout << "Double:" << p.first << "," << p.second << ":" << myMin << endl;

	return p;
}

int main()
{
	Mat image01 = imread("texture/rock.bmp");
	Mat targetImg = Mat(512, 512, CV_8UC3);
	Mat targetWithCut = Mat(targetImg.rows, targetImg.cols, CV_8UC3);
	Mat blendingTest = Mat(targetImg.rows, targetImg.cols, CV_8UC3);;


	const int patchN = 48;
	const int overlapPatchW = patchN / 6;
	const int patchOffset = patchN - overlapPatchW;
	const int patchRowNum = (targetImg.rows%patchOffset == 0) ? targetImg.rows / patchOffset : targetImg.rows / patchOffset + 1;
	const int patchColNum = (targetImg.cols%patchOffset == 0) ? targetImg.cols / patchOffset : targetImg.cols / patchOffset + 1;

	Mat boundaryTest = Mat(image01.rows * 4, image01.cols * 4, CV_8U);

	vector<pair<int, int>> sourcePatches;

	cout << "Input size:" << image01.rows << "," << image01.cols << endl;
	cout << "Target patch row number:" << patchRowNum << endl;
	cout << "Target patch column number:" << patchColNum << endl;
	cout << "Overlap patch size:" << overlapPatchW << endl;
	cout << "Patch offset:" << patchOffset << endl;
	cout << endl;

	srand(time(NULL));

	int randomX = rand() % (image01.rows - patchN);
	int randomY = rand() % (image01.cols - patchN);

	sourcePatches.push_back({ randomX, randomY });

	for (int ox = 0; ox < patchRowNum; ox++)
	{
		for (int oy = 0; oy < patchColNum; oy++)
		{
			Mat overlapRightPatch = Mat(patchN, overlapPatchW, CV_8UC3);
			Mat overlapDownPatch = Mat(overlapPatchW, patchN, CV_8UC3);

			if (ox == 0 && oy == 0)
				continue;

			// row1
			if (ox == 0)
			{
				for (int ix = 0; ix < overlapRightPatch.rows; ix++)
				{
					for (int iy = 0; iy < overlapRightPatch.cols; iy++)
					{
						overlapRightPatch.at<Vec3b>(ix, iy) = image01.at<Vec3b>(sourcePatches.back().first + ix, sourcePatches.back().second + iy + patchOffset);
					}
				}

				sourcePatches.push_back(SSD(image01, overlapRightPatch, patchN));
			}

			// col1
			if (ox > 0 && oy == 0)
			{
				for (int ix = 0; ix < overlapDownPatch.rows; ix++)
				{
					for (int iy = 0; iy < overlapDownPatch.cols; iy++)
					{
						overlapDownPatch.at<Vec3b>(ix, iy) = image01.at<Vec3b>(sourcePatches[sourcePatches.size() - patchColNum].first + ix + patchOffset,
							sourcePatches[sourcePatches.size() - patchColNum].second + iy);
					}
				}

				sourcePatches.push_back(SSD(image01, overlapDownPatch, patchN));
			}

			// except row1 and col1
			if (ox > 0 && oy > 0)
			{
				for (int ix = 0; ix < overlapRightPatch.rows; ix++)
				{
					for (int iy = 0; iy < overlapRightPatch.cols; iy++)
					{
						overlapRightPatch.at<Vec3b>(ix, iy) = image01.at<Vec3b>(sourcePatches.back().first + ix,
							sourcePatches.back().second + iy + patchOffset);
					}
				}

				for (int ix = 0; ix < overlapDownPatch.rows; ix++)
				{
					for (int iy = 0; iy < overlapDownPatch.cols; iy++)
					{
						overlapDownPatch.at<Vec3b>(ix, iy) = image01.at<Vec3b>(sourcePatches[sourcePatches.size() - patchColNum].first + ix + patchOffset,
							sourcePatches[sourcePatches.size() - patchColNum].second + iy);
					}
				}

				sourcePatches.push_back(BSSD(image01, overlapRightPatch, overlapDownPatch, patchN));
			}
		}

	}

	// minimum error boundary cut

	vector<vector<pair<int, int>>> allMebc(1, vector<pair<int, int>>(patchN, { 0, 0 }));
	vector<vector<vector<int>>> allOverlapCost(1, vector<vector<int>>(0, vector<int>(0, 0)));


	for (int i = 1; i < sourcePatches.size(); i++)
	{
		Mat overlapRightPatch = Mat(patchN, overlapPatchW, CV_8U);
		Mat overlapDownPatch = Mat(overlapPatchW, patchN, CV_8U);
		vector<vector<int>> rightCost(patchN, vector<int>(overlapPatchW, 0));
		vector<vector<int>> downCost(overlapPatchW, vector<int>(patchN, 0));

		if (i < patchColNum)
		{
			// distance map
			for (int ox = 0; ox < patchN; ox++)
			{
				for (int oy = 0; oy < overlapPatchW; oy++)
				{
					overlapRightPatch.at<uchar>(ox, oy) = sqrt(pow(image01.at<Vec3b>(sourcePatches[i - 1].first + ox, sourcePatches[i - 1].second + oy + patchOffset)[0]
						- image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[0], 2)

						+ pow(image01.at<Vec3b>(sourcePatches[i - 1].first + ox, sourcePatches[i - 1].second + oy + patchOffset)[1]
						- image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[1], 2)

						+ pow(image01.at<Vec3b>(sourcePatches[i - 1].first + ox, sourcePatches[i - 1].second + oy + patchOffset)[2]
						- image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[2], 2)) / sqrt(255.0*255.0*3.0)*255.0;

				}
			}

			// dp
			for (int ox = 0; ox < patchN; ox++)
			{
				for (int oy = 0; oy < overlapPatchW; oy++)
				{
					if (ox == 0)
						rightCost[ox][oy] = (int)overlapRightPatch.at<uchar>(ox, oy);
					else
					{
						if (oy == 0)
						{
							rightCost[ox][oy] = (int)overlapRightPatch.at<uchar>(ox, oy)
								+ min(rightCost[ox - 1][oy], rightCost[ox - 1][oy + 1]);
						}
						else if (oy == overlapPatchW - 1)
						{
							rightCost[ox][oy] = (int)overlapRightPatch.at<uchar>(ox, oy)
								+ min(rightCost[ox - 1][oy - 1], rightCost[ox - 1][oy]);
						}
						else
						{
							rightCost[ox][oy] = (int)overlapRightPatch.at<uchar>(ox, oy)
								+ min(min(rightCost[ox - 1][oy - 1], rightCost[ox - 1][oy]), rightCost[ox - 1][oy + 1]);
						}
					}
				}
			}

			allOverlapCost.push_back(rightCost);

			// trace back cut
			vector<pair<int, int>> mebc(patchN, { 0, 0 });

			for (int ox = patchN - 1; ox >= 0; ox--)
			{
				int x, y;

				if (ox == patchN - 1)
				{
					int myMin = INT_MAX;
					for (int oy = 0; oy < overlapPatchW; oy++)
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
						if (newOy < 0 || newOy >= overlapPatchW)
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
		else if (i%patchColNum == 0)
		{

			// distance map
			for (int ox = 0; ox < overlapPatchW; ox++)
			{
				for (int oy = 0; oy < patchN; oy++)
				{
					overlapDownPatch.at<uchar>(ox, oy) = sqrt(pow(image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[0]
						- image01.at<Vec3b>(sourcePatches[i - patchColNum].first + ox + patchOffset, sourcePatches[i - patchColNum].second + oy)[0], 2)

						+ pow(image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[1]
						- image01.at<Vec3b>(sourcePatches[i - patchColNum].first + ox + patchOffset, sourcePatches[i - patchColNum].second + oy)[1], 2)

						+ pow(image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[2]
						- image01.at<Vec3b>(sourcePatches[i - patchColNum].first + ox + patchOffset, sourcePatches[i - patchColNum].second + oy)[2], 2)) / sqrt(255.0*255.0*3.0)*255.0;

				}
			}

			// dp
			for (int oy = 0; oy < patchN; oy++)
			{
				for (int ox = 0; ox < overlapPatchW; ox++)
				{
					if (oy == 0)
						downCost[ox][oy] = (int)overlapDownPatch.at<uchar>(ox, oy);
					else
					{
						if (ox == 0)
						{
							downCost[ox][oy] = (int)overlapDownPatch.at<uchar>(ox, oy)
								+ min(downCost[ox][oy - 1], downCost[ox + 1][oy - 1]);
						}
						else if (ox == overlapPatchW - 1)
						{
							downCost[ox][oy] = (int)overlapDownPatch.at<uchar>(ox, oy)
								+ min(downCost[ox - 1][oy - 1], downCost[ox][oy - 1]);
						}
						else
						{
							downCost[ox][oy] = (int)overlapDownPatch.at<uchar>(ox, oy)
								+ min(min(downCost[ox - 1][oy - 1], downCost[ox][oy - 1]), downCost[ox + 1][oy - 1]);
						}
					}
				}
			}

			allOverlapCost.push_back(downCost);

			// trace back cut
			vector<pair<int, int>> mebc(patchN, { 0, 0 });

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
		else
		{
			// distance map
			for (int ox = 0; ox < patchN; ox++)
			{
				for (int oy = 0; oy < overlapPatchW; oy++)
				{
					overlapRightPatch.at<uchar>(ox, oy) = sqrt(pow(image01.at<Vec3b>(sourcePatches[i - 1].first + ox, sourcePatches[i - 1].second + oy + patchOffset)[0]
						- image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[0], 2)

						+ pow(image01.at<Vec3b>(sourcePatches[i - 1].first + ox, sourcePatches[i - 1].second + oy + patchOffset)[1]
						- image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[1], 2)

						+ pow(image01.at<Vec3b>(sourcePatches[i - 1].first + ox, sourcePatches[i - 1].second + oy + patchOffset)[2]
						- image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[2], 2)) / sqrt(255.0*255.0*3.0)*255.0;

				}
			}

			// dp
			for (int ox = 0; ox < patchN; ox++)
			{
				for (int oy = 0; oy < overlapPatchW; oy++)
				{
					if (ox == 0)
						rightCost[ox][oy] = (int)overlapRightPatch.at<uchar>(ox, oy);
					else
					{
						if (oy == 0)
						{
							rightCost[ox][oy] = (int)overlapRightPatch.at<uchar>(ox, oy)
								+ min(rightCost[ox - 1][oy], rightCost[ox - 1][oy + 1]);
						}
						else if (oy == overlapPatchW - 1)
						{
							rightCost[ox][oy] = (int)overlapRightPatch.at<uchar>(ox, oy)
								+ min(rightCost[ox - 1][oy - 1], rightCost[ox - 1][oy]);
						}
						else
						{
							rightCost[ox][oy] = (int)overlapRightPatch.at<uchar>(ox, oy)
								+ min(min(rightCost[ox - 1][oy - 1], rightCost[ox - 1][oy]), rightCost[ox - 1][oy + 1]);
						}
					}
				}
			}

			allOverlapCost.push_back(rightCost);

			// trace back cut
			vector<pair<int, int>> mebc(patchN, { 0, 0 });

			for (int ox = patchN - 1; ox >= 0; ox--)
			{

				int x, y;

				if (ox == patchN - 1)
				{
					int myMin = INT_MAX;
					for (int oy = 0; oy < overlapPatchW; oy++)
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
						if (newOy < 0 || newOy >= overlapPatchW)
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

			// distance map
			for (int ox = 0; ox < overlapPatchW; ox++)
			{
				for (int oy = 0; oy < patchN; oy++)
				{
					overlapDownPatch.at<uchar>(ox, oy) = sqrt(pow(image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[0]
						- image01.at<Vec3b>(sourcePatches[i - patchColNum].first + ox + patchOffset, sourcePatches[i - patchColNum].second + oy)[0], 2)

						+ pow(image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[1]
						- image01.at<Vec3b>(sourcePatches[i - patchColNum].first + ox + patchOffset, sourcePatches[i - patchColNum].second + oy)[1], 2)

						+ pow(image01.at<Vec3b>(sourcePatches[i].first + ox, sourcePatches[i].second + oy)[2]
						- image01.at<Vec3b>(sourcePatches[i - patchColNum].first + ox + patchOffset, sourcePatches[i - patchColNum].second + oy)[2], 2)) / sqrt(255.0*255.0*3.0)*255.0;

				}
			}

			// dp
			for (int oy = 0; oy < patchN; oy++)
			{
				for (int ox = 0; ox < overlapPatchW; ox++)
				{
					if (oy == 0)
						downCost[ox][oy] = (int)overlapDownPatch.at<uchar>(ox, oy);
					else
					{
						if (ox == 0)
						{
							downCost[ox][oy] = (int)overlapDownPatch.at<uchar>(ox, oy)
								+ min(downCost[ox][oy - 1], downCost[ox + 1][oy - 1]);
						}
						else if (ox == overlapPatchW - 1)
						{
							downCost[ox][oy] = (int)overlapDownPatch.at<uchar>(ox, oy)
								+ min(downCost[ox - 1][oy - 1], downCost[ox][oy - 1]);
						}
						else
						{
							downCost[ox][oy] = (int)overlapDownPatch.at<uchar>(ox, oy)
								+ min(min(downCost[ox - 1][oy - 1], downCost[ox][oy - 1]), downCost[ox + 1][oy - 1]);
						}
					}
				}
			}

			allOverlapCost.push_back(downCost);

			// trace back cut
			mebc = vector<pair<int, int>>(patchN, { 0, 0 });

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
	}

	// Boundary cut test
	Mat test = Mat(patchN, overlapPatchW, CV_8U);
	Mat test2 = Mat(patchN, overlapPatchW, CV_8U);

	/*for (int ox = 0; ox < patchN; ox++)
	{
		for (int oy = 0; oy < overlapPatchW; oy++)
		{
			test.at<uchar>(ox, oy) = sqrt(pow(image01.at<Vec3b>(sourcePatches[16].first + ox, sourcePatches[16].second + oy + patchOffset)[0]
				- image01.at<Vec3b>(sourcePatches[17].first + ox, sourcePatches[17].second + oy)[0], 2)

				+ pow(image01.at<Vec3b>(sourcePatches[16].first + ox, sourcePatches[16].second + oy + patchOffset)[1]
				- image01.at<Vec3b>(sourcePatches[17].first + ox, sourcePatches[17].second + oy)[1], 2)

				+ pow(image01.at<Vec3b>(sourcePatches[16].first + ox, sourcePatches[16].second + oy + patchOffset)[2]
				- image01.at<Vec3b>(sourcePatches[17].first + ox, sourcePatches[17].second + oy)[2], 2)) / sqrt(255.0*255.0*3.0)*255.0;

			test2.at<uchar>(ox, oy) = test.at<uchar>(ox, oy);

			if (ox == allMebc[20][ox].first && oy == allMebc[20][ox].second)
				test.at<uchar>(ox, oy) = 255;
		}
	}*/

	// quilting along cut

	int patchIndex = 0;
	int boundaryIndex = 0;

	for (int ox = 0; ox < targetImg.rows; ox += patchOffset)
	{
		for (int oy = 0; oy < targetImg.cols; oy += patchOffset)
		{
			// row1
			if (patchIndex < patchColNum)
			{
				for (int ix = 0; ix < patchN; ix++)
				{
					if (ox + ix >= targetImg.rows)
						break;

					if (patchIndex == 0)
					{
						for (int iy = 0; iy < patchN; iy++)
						{
							if (oy + iy >= targetImg.cols)
								break;
							targetWithCut.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
							targetImg.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
							blendingTest.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
						}
					}
					else
					{
						for (int iy = allMebc[boundaryIndex][ix].second; iy < patchN; iy++)
						{
							if (oy + iy >= targetImg.cols)
								break;
							targetWithCut.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
							targetImg.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);

							if (iy == allMebc[boundaryIndex][ix].second)
							{
								targetWithCut.at<Vec3b>(ox + ix, oy + iy) = Vec3b(255, 255, 255);

								Vec3f a = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
								Vec3f b = blendingTest.at<Vec3b>(ox + ix, oy + iy);
								Vec3f c = (a + b)*0.5;

								blendingTest.at<Vec3b>(ox + ix, oy + iy) = c;
							}
							else
							{
								blendingTest.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
							}

						}
					}
				}
				patchIndex++;
				boundaryIndex++;
			}
			else if (patchIndex % patchColNum == 0)	// col1
			{
				for (int iy = 0; iy < patchN; iy++)
				{
					for (int ix = allMebc[boundaryIndex][iy].first; ix < patchN; ix++)
					{
						if (ox + ix >= targetImg.rows)
							break;

						// row2
						if (patchIndex == patchColNum)
						{
							if (ix + patchN - overlapPatchW < patchN)
							{			
													//上一個row的right cut
								if (iy >= allMebc[boundaryIndex - (patchColNum - 1)][ix + patchN - overlapPatchW].second + patchN - overlapPatchW)
								{
									continue;
								}
							}
						}
						else // row3 to rowN
						{
							if (ix + patchN - overlapPatchW < patchN)
							{
													//上一個row的right cut
								if (iy >= allMebc[boundaryIndex - (patchColNum - 1) * 2][ix + patchN - overlapPatchW].second + patchN - overlapPatchW)
								{
									continue;
								}
							}
						}

						targetWithCut.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
						targetImg.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);

						if (ix == allMebc[boundaryIndex][iy].first)
						{
							targetWithCut.at<Vec3b>(ox + ix, oy + iy) = Vec3b(255, 255, 255);

							Vec3f a = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
							Vec3f b = blendingTest.at<Vec3b>(ox + ix, oy + iy);
							Vec3f c = (a + b)*0.5;

							blendingTest.at<Vec3b>(ox + ix, oy + iy) = c;
						}
						else
						{
							blendingTest.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
						}

					}
				}
				patchIndex++;
				boundaryIndex++;
			}
			else // except row1 and col1
			{
				// top down
				for (int ix = 0; ix < patchN; ix++)
				{
					if (ox + ix >= targetImg.rows)
						break;

					// 把cut當成起點(包含)
					for (int iy = allMebc[boundaryIndex][ix].second; iy < patchN; iy++)
					{
						if (oy + iy >= targetImg.cols)
							break;

						// row2
						if (patchIndex < patchColNum * 2)
						{
							if (ix + patchN - overlapPatchW < patchN && iy + patchN - overlapPatchW < patchN)
							{
								// 小於上一row的right cut且高於左邊的top cut不畫
								if (iy < allMebc[boundaryIndex - patchIndex + 1][ix + patchN - overlapPatchW].second && ix < allMebc[boundaryIndex - 1][iy + patchN - overlapPatchW].first)
								{
									continue;
								}
							}

							if (ix + patchN - overlapPatchW < patchN)
							{
								// 小於上一row的下一個right cut
								if (iy >= allMebc[boundaryIndex - patchIndex + 2][ix + patchN - overlapPatchW].second + patchN - overlapPatchW)
								{
									continue;
								}
							}


						}
						else // row3 to rowN
						{
							if (ix + patchN - overlapPatchW < patchN && iy + patchN - overlapPatchW < patchN)
							{
								if (iy < allMebc[boundaryIndex - 2 * (patchColNum - 1) - 1][ix + patchN - overlapPatchW].second && ix < allMebc[boundaryIndex - 1][iy + patchN - overlapPatchW].first)
								{
									continue;
								}
							}

							if (ix + patchN - overlapPatchW < patchN)
							{

								if (iy >= allMebc[boundaryIndex - 2 * (patchColNum - 1) - 1 + 2][ix + patchN - overlapPatchW].second + patchN - overlapPatchW)
								{
									continue;
								}
							}
						}

						// 超過top cut不覆蓋
						if (ix < allMebc[boundaryIndex + 1][iy].first)
						{
							continue;
						}

						targetWithCut.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
						targetImg.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);

						if (iy == allMebc[boundaryIndex][ix].second)
						{
							targetWithCut.at<Vec3b>(ox + ix, oy + iy) = Vec3b(255, 255, 255);

							Vec3f a = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
							Vec3f b = blendingTest.at<Vec3b>(ox + ix, oy + iy);
							Vec3f c = (a + b)*0.5;

							blendingTest.at<Vec3b>(ox + ix, oy + iy) = c;

						}
						else if (ix == allMebc[boundaryIndex + 1][iy].first)
						{
							targetWithCut.at<Vec3b>(ox + ix, oy + iy) = Vec3b(255, 255, 255);

							Vec3f a = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
							Vec3f b = blendingTest.at<Vec3b>(ox + ix, oy + iy);
							Vec3f c = (a + b)*0.5;

							blendingTest.at<Vec3b>(ox + ix, oy + iy) = c;
						}
						else
						{
							blendingTest.at<Vec3b>(ox + ix, oy + iy) = image01.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
						}
						
					}
				}

				patchIndex++;
				boundaryIndex += 2;
			}
		}
	}

	imshow("image01", image01);
	imshow("targetImg", targetImg);
	imshow("targetWithCut", targetWithCut);
	imshow("blendingTest", blendingTest);
	imshow("boundaryWithCut", test);
	imshow("boundary", test2);

	waitKey(0);
}