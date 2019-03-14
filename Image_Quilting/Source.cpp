#include <iostream>
#include "opencv2\opencv.hpp"
#include "opencv2\nonfree\nonfree.hpp"

#include <math.h>
#include <time.h>
#include <vector>
#include "Distance.h"
#include "BoundaryCut.h"

using namespace std;
using namespace cv;

int main()
{
	time_t s = time(NULL);

	bool imgQuliting = 0, texTransfer = 1;

	Mat srcImg = imread("texture/rice.bmp");
	Mat targetImg = imread("texture/bill-big-big.jpg");
	Mat outputImg;

	if (imgQuliting)
		outputImg = Mat(512, 512, CV_8UC3);
	else if (texTransfer)
		outputImg = Mat(targetImg.rows, targetImg.cols, CV_8UC3);

	Mat outputWithCut = Mat(outputImg.rows, outputImg.cols, CV_8UC3);

	vector<Mat> results;

	int patchN = 18;
	int overlapPatchW = patchN / 6;
	int patchOffset = patchN - overlapPatchW;
	int patchRowNum = (outputImg.rows % patchOffset == 0) ? outputImg.rows / patchOffset : outputImg.rows / patchOffset + 1;
	int patchColNum = (outputImg.cols % patchOffset == 0) ? outputImg.cols / patchOffset : outputImg.cols / patchOffset + 1;

	vector<pair<int, int>> sourcePatches;

	Mat srcImgHSV;
	cvtColor(srcImg, srcImgHSV, CV_BGR2HSV);
	vector<Mat> srcChannels;
	split(srcImgHSV, srcChannels);

	int iteration;

	if (texTransfer)
		iteration = 1;
	else if (imgQuliting)
		iteration = 1;

	for (int N = 0; N < iteration; N++)
	{
		cout << "Iteration:" << N << endl;

		if (N > 0)
		{
			targetImg = outputImg;
			patchN = patchN * sqrt(0.6666667);
			overlapPatchW = patchN / 6;
			patchOffset = patchN - overlapPatchW;
			patchRowNum = (outputImg.rows % patchOffset == 0) ? outputImg.rows / patchOffset : outputImg.rows / patchOffset + 1;
			patchColNum = (outputImg.cols % patchOffset == 0) ? outputImg.cols / patchOffset : outputImg.cols / patchOffset + 1;
		}

		
		Mat targetImgHSV;
		cvtColor(targetImg, targetImgHSV, CV_BGR2HSV);
		vector<Mat> targetChannels;
		split(targetImgHSV, targetChannels);

		cout << "Source size:" << srcImg.rows << "," << srcImg.cols << endl;
		cout << "Target size:" << targetImg.rows << "," << targetImg.cols << endl;
		cout << "Target patch row number:" << patchRowNum << endl;
		cout << "Target patch column number:" << patchColNum << endl;
		cout << "Overlap patch size:" << overlapPatchW << endl;
		cout << "Patch offset:" << patchOffset << endl;
		cout << endl;

		vector<pair<int, int>> sourcePatches2;

		
		float a;

		if (N < 2)
			a = 0.3;
		else
			a = 0.8 * (N - 1) / (iteration - 1) + 0.1;

		if (texTransfer)
		{
			// NNF
			for (int ox = 0; ox < patchRowNum; ox++)
			{
				for (int oy = 0; oy < patchColNum; oy++)
				{
					int squarePatchRow = (patchOffset * ox + patchN < outputImg.rows) ? patchN : (outputImg.rows - (patchOffset * ox));
					int squarePatchCol = (patchOffset * oy + patchN < outputImg.cols) ? patchN : (outputImg.cols - (patchOffset * oy));


					Mat overlapRightPatch = Mat(patchN, overlapPatchW, CV_8UC3);
					Mat overlapDownPatch = Mat(overlapPatchW, patchN, CV_8UC3);
					Mat overlapSquarePatch = Mat(squarePatchRow, squarePatchCol, CV_8U);
					Distance d;

					if (ox == 0 && oy == 0)
					{
						for (int ix = 0; ix < overlapSquarePatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapSquarePatch.cols; iy++)
							{
								overlapSquarePatch.at<uchar>(ix, iy) = targetChannels.at(2).at<uchar>(ix, iy);
								//cout << (float)targetChannels.at(2).at<uchar>(ix, iy) << endl;
							}
						}
						cout << ox << "," << oy << endl;
						sourcePatches2.push_back(d.LSSD(srcChannels.at(2), overlapSquarePatch, patchN));

						continue;
					}

					// row1
					if (ox == 0)
					{
						for (int ix = 0; ix < overlapRightPatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapRightPatch.cols; iy++)
							{
								overlapRightPatch.at<Vec3b>(ix, iy) = srcImg.at<Vec3b>(sourcePatches2.back().first + ix, sourcePatches2.back().second + iy + patchOffset);
							}
						}

						for (int ix = 0; ix < overlapSquarePatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapSquarePatch.cols; iy++)
							{
								overlapSquarePatch.at<uchar>(ix, iy) = targetChannels.at(2).at<uchar>(ix + patchOffset * ox, iy + patchOffset * oy);
								//cout << overlapSquarePatch.at<uchar>(ix, iy) << endl;
							}
						}

						cout << ox << "," << oy << endl;
						sourcePatches2.push_back(d.LAndSSD(srcImg, overlapRightPatch, srcChannels.at(2), overlapSquarePatch, patchN, a));
					}

					// col1
					if (ox > 0 && oy == 0)
					{
						for (int ix = 0; ix < overlapDownPatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapDownPatch.cols; iy++)
							{
								overlapDownPatch.at<Vec3b>(ix, iy) = srcImg.at<Vec3b>(sourcePatches2[sourcePatches2.size() - patchColNum].first + ix + patchOffset,
									sourcePatches2[sourcePatches2.size() - patchColNum].second + iy);
							}
						}

						for (int ix = 0; ix < overlapSquarePatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapSquarePatch.cols; iy++)
							{
								overlapSquarePatch.at<uchar>(ix, iy) = targetChannels.at(2).at<uchar>(ix + patchOffset * ox, iy + patchOffset * oy);
								//cout << overlapSquarePatch.at<uchar>(ix, iy) << endl;
							}
						}
						cout << ox << "," << oy << endl;
						sourcePatches2.push_back(d.LAndSSD(srcImg, overlapDownPatch, srcChannels.at(2), overlapSquarePatch, patchN, a));
					}

					// except row1 and col1
					if (ox > 0 && oy > 0)
					{
						for (int ix = 0; ix < overlapRightPatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapRightPatch.cols; iy++)
							{
								overlapRightPatch.at<Vec3b>(ix, iy) = srcImg.at<Vec3b>(sourcePatches2.back().first + ix,
									sourcePatches2.back().second + iy + patchOffset);
							}
						}

						for (int ix = 0; ix < overlapDownPatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapDownPatch.cols; iy++)
							{
								overlapDownPatch.at<Vec3b>(ix, iy) = srcImg.at<Vec3b>(sourcePatches2[sourcePatches2.size() - patchColNum].first + ix + patchOffset,
									sourcePatches2[sourcePatches2.size() - patchColNum].second + iy);
							}
						}

						for (int ix = 0; ix < overlapSquarePatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapSquarePatch.cols; iy++)
							{
								overlapSquarePatch.at<uchar>(ix, iy) = targetChannels.at(2).at<uchar>(ix + patchOffset * ox, iy + patchOffset * oy);
								//cout << overlapSquarePatch.at<uchar>(ix, iy) << endl;
							}
						}
						cout << ox << "," << oy << endl;


						sourcePatches2.push_back(d.LAndBSSD(srcImg, overlapRightPatch, overlapDownPatch, srcChannels.at(2), overlapSquarePatch, patchN, a));
					}
				}

			}

			sourcePatches = sourcePatches2;

		}
		else if (imgQuliting)
		{
			// NNF
			srand(time(NULL));

			int randomX = rand() % (srcImg.rows - patchN);
			int randomY = rand() % (srcImg.cols - patchN);

			sourcePatches.push_back({ randomX, randomY });

			for (int ox = 0; ox < patchRowNum; ox++)
			{
				for (int oy = 0; oy < patchColNum; oy++)
				{
					Mat overlapRightPatch = Mat(patchN, overlapPatchW, CV_8UC3);
					Mat overlapDownPatch = Mat(overlapPatchW, patchN, CV_8UC3);
					Distance d;

					if (ox == 0 && oy == 0)
						continue;

					// row1
					if (ox == 0)
					{
						for (int ix = 0; ix < overlapRightPatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapRightPatch.cols; iy++)
							{
								overlapRightPatch.at<Vec3b>(ix, iy) = srcImg.at<Vec3b>(sourcePatches.back().first + ix, sourcePatches.back().second + iy + patchOffset);
							}
						}

						sourcePatches.push_back(d.SSD(srcImg, overlapRightPatch, patchN));
					}

					// col1
					if (ox > 0 && oy == 0)
					{
						for (int ix = 0; ix < overlapDownPatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapDownPatch.cols; iy++)
							{
								overlapDownPatch.at<Vec3b>(ix, iy) = srcImg.at<Vec3b>(sourcePatches[sourcePatches.size() - patchColNum].first + ix + patchOffset,
									sourcePatches[sourcePatches.size() - patchColNum].second + iy);
							}
						}

						sourcePatches.push_back(d.SSD(srcImg, overlapDownPatch, patchN));
					}

					// except row1 and col1
					if (ox > 0 && oy > 0)
					{
						for (int ix = 0; ix < overlapRightPatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapRightPatch.cols; iy++)
							{
								overlapRightPatch.at<Vec3b>(ix, iy) = srcImg.at<Vec3b>(sourcePatches.back().first + ix,
									sourcePatches.back().second + iy + patchOffset);
							}
						}

						for (int ix = 0; ix < overlapDownPatch.rows; ix++)
						{
							for (int iy = 0; iy < overlapDownPatch.cols; iy++)
							{
								overlapDownPatch.at<Vec3b>(ix, iy) = srcImg.at<Vec3b>(sourcePatches[sourcePatches.size() - patchColNum].first + ix + patchOffset,
									sourcePatches[sourcePatches.size() - patchColNum].second + iy);
							}
						}

						sourcePatches.push_back(d.BSSD(srcImg, overlapRightPatch, overlapDownPatch, patchN));
					}
				}

			}
		}

		// minimum error boundary cut

		vector<vector<pair<int, int>>> allMebc(1, vector<pair<int, int>>(patchN, { 0, 0 }));

		for (int i = 1; i < sourcePatches.size(); i++)
		{
			Mat overlapRightPatch = Mat(patchN, overlapPatchW, CV_8U);
			Mat overlapDownPatch = Mat(overlapPatchW, patchN, CV_8U);
			vector<vector<int>> rightCost(patchN, vector<int>(overlapPatchW, 0));
			vector<vector<int>> downCost(overlapPatchW, vector<int>(patchN, 0));
			BoundaryCut bc;

			if (i < patchColNum)
			{
				bc.computeRightCut(patchOffset, overlapRightPatch, allMebc, srcImg, sourcePatches, i);
			}
			else if (i%patchColNum == 0)
			{
				bc.computeDownCut(patchOffset, overlapDownPatch, allMebc, srcImg, sourcePatches, i, patchColNum);
			}
			else
			{
				bc.computeRightCut(patchOffset, overlapRightPatch, allMebc, srcImg, sourcePatches, i);
				bc.computeDownCut(patchOffset, overlapDownPatch, allMebc, srcImg, sourcePatches, i, patchColNum);
			}
		}

		// quilting along cut

		int patchIndex = 0;
		int boundaryIndex = 0;

		for (int ox = 0; ox < outputImg.rows; ox += patchOffset)
		{
			for (int oy = 0; oy < outputImg.cols; oy += patchOffset)
			{
				// row1
				if (patchIndex < patchColNum)
				{
					for (int ix = 0; ix < patchN; ix++)
					{
						if (ox + ix >= outputImg.rows)
							break;

						if (patchIndex == 0)
						{
							for (int iy = 0; iy < patchN; iy++)
							{
								if (oy + iy >= outputImg.cols)
									break;
								outputWithCut.at<Vec3b>(ox + ix, oy + iy) = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
								outputImg.at<Vec3b>(ox + ix, oy + iy) = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
							}
						}
						else
						{
							for (int iy = allMebc[boundaryIndex][ix].second; iy < patchN; iy++)
							{
								if (oy + iy >= outputImg.cols)
									break;
								outputWithCut.at<Vec3b>(ox + ix, oy + iy) = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);

								if (iy == allMebc[boundaryIndex][ix].second)
								{
									outputWithCut.at<Vec3b>(ox + ix, oy + iy) = Vec3b(255, 255, 255);

									Vec3f a = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
									Vec3f b = outputImg.at<Vec3b>(ox + ix, oy + iy);
									Vec3f c = (a + b)*0.5;

									outputImg.at<Vec3b>(ox + ix, oy + iy) = c;
								}
								else
								{
									outputImg.at<Vec3b>(ox + ix, oy + iy) = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
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
							if (ox + ix >= outputImg.rows)
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

							outputWithCut.at<Vec3b>(ox + ix, oy + iy) = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);

							if (ix == allMebc[boundaryIndex][iy].first)
							{
								outputWithCut.at<Vec3b>(ox + ix, oy + iy) = Vec3b(255, 255, 255);

								Vec3f a = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
								Vec3f b = outputImg.at<Vec3b>(ox + ix, oy + iy);
								Vec3f c = (a + b)*0.5;

								outputImg.at<Vec3b>(ox + ix, oy + iy) = c;
							}
							else
							{
								outputImg.at<Vec3b>(ox + ix, oy + iy) = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
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
						if (ox + ix >= outputImg.rows)
							break;

						// 把cut當成起點(包含)
						for (int iy = allMebc[boundaryIndex][ix].second; iy < patchN; iy++)
						{
							if (oy + iy >= outputImg.cols)
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

							outputWithCut.at<Vec3b>(ox + ix, oy + iy) = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);

							if (iy == allMebc[boundaryIndex][ix].second)
							{
								outputWithCut.at<Vec3b>(ox + ix, oy + iy) = Vec3b(255, 255, 255);

								Vec3f a = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
								Vec3f b = outputImg.at<Vec3b>(ox + ix, oy + iy);
								Vec3f c = (a + b)*0.5;

								outputImg.at<Vec3b>(ox + ix, oy + iy) = c;

							}
							else if (ix == allMebc[boundaryIndex + 1][iy].first)
							{
								outputWithCut.at<Vec3b>(ox + ix, oy + iy) = Vec3b(255, 255, 255);

								Vec3f a = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
								Vec3f b = outputImg.at<Vec3b>(ox + ix, oy + iy);
								Vec3f c = (a + b)*0.5;

								outputImg.at<Vec3b>(ox + ix, oy + iy) = c;
							}
							else
							{
								outputImg.at<Vec3b>(ox + ix, oy + iy) = srcImg.at<Vec3b>(sourcePatches[patchIndex].first + ix, sourcePatches[patchIndex].second + iy);
							}

						}
					}

					patchIndex++;
					boundaryIndex += 2;
				}
			}
		}

		results.push_back(outputImg.clone());
	}

	imshow("srcImg", srcImg);
	imshow("targetImg", targetImg);
	imshow("outputImg", outputImg);
	imshow("outputWithCut", outputWithCut);

	for (int i = 0; i < results.size(); i++)
	{
		string name = "Iteration" + to_string(i);
		imshow(name, results[i]);
	}

	time_t e = time(NULL);

	cout << "Time:" << (e - s) / 60 << "m" << endl;

	waitKey(0);
}