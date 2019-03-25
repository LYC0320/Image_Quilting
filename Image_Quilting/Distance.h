#include "opencv2\opencv.hpp"
#include "opencv2\nonfree\nonfree.hpp"

using namespace std;
using namespace cv;

class Distance
{
public:
	Distance();
	~Distance();

	pair<int, int> SSD(Mat s, Mat t, int patchN);
	pair<int, int> BSSD(Mat s, Mat t1, Mat t2, int patchN);
	pair<int, int> LSSD(Mat s, Mat t, int patchN);
	pair<int, int> LAndSSD(Mat s1, Mat t1, Mat s2, Mat t2, int patchN, float a);
	pair<int, int> LAndBSSD(Mat s1, Mat t1, Mat t2, Mat s2, Mat t3, int patchN, float a);

private:

	float et = 1.2;

};

Distance::Distance()
{
}

Distance::~Distance()
{
}

pair<int, int> Distance::SSD(Mat s, Mat t, int patchN)
{
	double myMin;
	Point minLoc;
	pair<int, int> p;

	vector<Mat> sC, tC;
	split(s, sC);
	split(t, tC);

	Mat tmp0, tmp1, tmp2 = Mat(s.rows - t.rows + 1, s.cols - t.cols + 1, CV_32F);

	Mat tmpAll = Mat(s.rows - patchN, s.cols - patchN, CV_32F);

	matchTemplate(sC.at(0), tC.at(0), tmp0, CV_TM_SQDIFF);
	matchTemplate(sC.at(1), tC.at(1), tmp1, CV_TM_SQDIFF);
	matchTemplate(sC.at(2), tC.at(2), tmp2, CV_TM_SQDIFF);

	sqrt(tmp0, tmp0);
	sqrt(tmp1, tmp1);
	sqrt(tmp2, tmp2);

	add(tmp0, tmp1, tmp0);
	add(tmp0, tmp2, tmp0);

	for (int ox = 0; ox < s.rows - patchN; ox++)
	{
		for (int oy = 0; oy < s.cols - patchN; oy++)
		{
			tmpAll.at<float>(ox, oy) = tmp0.at<float>(ox, oy);
		}
	}

	minMaxLoc(tmpAll, &myMin, 0, &minLoc, 0);
	p = { minLoc.y, minLoc.x };

	cout << "Single:" << p.first << "," << p.second << ":" << myMin << endl;

	return p;
}

pair<int, int> Distance::BSSD(Mat s, Mat t1, Mat t2, int patchN)
{
	double myMin;
	Point minLoc;
	pair<int, int> p;

	vector<Mat> sC, t1C, t2C;
	split(s, sC);
	split(t1, t1C);
	split(t2, t2C);

	Mat tmpT1_0, tmpT1_1, tmpT1_2 = Mat(s.rows - t1.rows + 1, s.cols - t1.cols + 1, CV_32F);
	Mat tmpT2_0, tmpT2_1, tmpT2_2 = Mat(s.rows - t2.rows + 1, s.cols - t2.cols + 1, CV_32F);

	Mat tmpAll = Mat(s.rows - patchN, s.cols - patchN, CV_32F);

	matchTemplate(sC.at(0), t1C.at(0), tmpT1_0, CV_TM_SQDIFF);
	matchTemplate(sC.at(1), t1C.at(1), tmpT1_1, CV_TM_SQDIFF);
	matchTemplate(sC.at(2), t1C.at(2), tmpT1_2, CV_TM_SQDIFF);

	sqrt(tmpT1_0, tmpT1_0);
	sqrt(tmpT1_1, tmpT1_1);
	sqrt(tmpT1_2, tmpT1_2);

	add(tmpT1_0, tmpT1_1, tmpT1_0);
	add(tmpT1_0, tmpT1_2, tmpT1_0);

	matchTemplate(sC.at(0), t2C.at(0), tmpT2_0, CV_TM_SQDIFF);
	matchTemplate(sC.at(1), t2C.at(1), tmpT2_1, CV_TM_SQDIFF);
	matchTemplate(sC.at(2), t2C.at(2), tmpT2_2, CV_TM_SQDIFF);

	sqrt(tmpT2_0, tmpT2_0);
	sqrt(tmpT2_1, tmpT2_1);
	sqrt(tmpT2_2, tmpT2_2);

	add(tmpT2_0, tmpT2_1, tmpT2_0);
	add(tmpT2_0, tmpT2_2, tmpT2_0);

	for (int ox = 0; ox < s.rows - patchN; ox++)
	{
		for (int oy = 0; oy < s.cols - patchN; oy++)
		{
			tmpAll.at<float>(ox, oy) = tmpT1_0.at<float>(ox, oy) + tmpT2_0.at<float>(ox, oy);
		}
	}

	minMaxLoc(tmpAll, &myMin, 0, &minLoc, 0);

	p = { minLoc.y, minLoc.x };

	cout << "Double:" << p.first << "," << p.second << ":" << myMin << endl;

	return p;
}

pair<int, int> Distance::LSSD(Mat s, Mat t, int patchN)
{
	pair<int, int> p;
	vector<pair<int, int>> matchingIndex;

	Mat tmp = Mat(s.rows - t.rows + 1, s.cols - t.cols + 1, CV_32F);
	double myMin;
	Point minLoc;
	matchTemplate(s, t, tmp, CV_TM_SQDIFF);
	sqrt(tmp, tmp);
	minMaxLoc(tmp, &myMin, 0, &minLoc, 0);

	// error tolerance
	for (int ox = 0; ox < s.rows - patchN; ox++)
	{
		for (int oy = 0; oy < s.cols - patchN; oy++)
		{
			if (tmp.at<float>(ox, oy) <= et*myMin)
			{
				matchingIndex.push_back({ ox, oy });
			}
		}
	}

	srand(time(NULL));

	p = matchingIndex[rand() % matchingIndex.size()];

	cout << "Single(L):" << p.first << "," << p.second << ":" << myMin << endl;

	return p;
}


pair<int, int> Distance::LAndSSD(Mat s1, Mat t1, Mat s2, Mat t2, int patchN, float a)
{
	vector<Mat> s1C, t1C;
	split(s1, s1C);
	split(t1, t1C);

	double myMin;
	Point minLoc;
	Mat tmp0, tmp1, tmp2 = Mat(s1.rows - t1.rows + 1, s1.cols - t1.cols + 1, CV_32F);

	Mat tmpAll = Mat(s1.rows - patchN, s1.cols - patchN, CV_32F);

	matchTemplate(s1C.at(0), t1C.at(0), tmp0, CV_TM_SQDIFF);
	matchTemplate(s1C.at(1), t1C.at(1), tmp1, CV_TM_SQDIFF);
	matchTemplate(s1C.at(2), t1C.at(2), tmp2, CV_TM_SQDIFF);


	sqrt(tmp0, tmp0);
	sqrt(tmp1, tmp1);
	sqrt(tmp2, tmp2);


	add(tmp0, tmp1, tmp0);
	add(tmp0, tmp2, tmp0);
	tmp0 = tmp0*0.3333333*a;


	Mat tmp3 = Mat(s2.rows - t2.rows + 1, s2.cols - t2.cols + 1, CV_32F);
	matchTemplate(s2, t2, tmp3, CV_TM_SQDIFF);
	sqrt(tmp3, tmp3);
	tmp3 = tmp3*(1 - a);

	for (int ox = 0; ox < s1.rows - patchN; ox++)
	{
		for (int oy = 0; oy < s1.cols - patchN; oy++)
		{
			tmpAll.at<float>(ox, oy) = tmp3.at<float>(ox, oy) + tmp0.at<float>(ox, oy);
		}
	}

	minMaxLoc(tmpAll, &myMin, 0, &minLoc, 0);

	pair<int, int> p;
	vector<pair<int, int>> matchingIndex;


	// error tolerance
	for (int ox = 0; ox < s1.rows - patchN; ox++)
	{
		for (int oy = 0; oy < s1.cols - patchN; oy++)
		{
			if (tmpAll.at<float>(ox, oy) <= et*myMin)
			{
				matchingIndex.push_back({ ox, oy });
			}
		}
	}

	srand(time(NULL));

	p = matchingIndex[rand() % matchingIndex.size()];

	cout << "Single(LAndSSD):" << p.first << "," << p.second << ":" << myMin << endl;

	return p;
}

pair<int, int> Distance::LAndBSSD(Mat s1, Mat t1, Mat t2, Mat s2, Mat t3, int patchN, float a)
{

	double myMin;
	Point minLoc;
	vector<pair<int, int>> matchingIndex;
	pair<int, int> p;

	vector<Mat> s1C, t1C, t2C;
	split(s1, s1C);
	split(t1, t1C);
	split(t2, t2C);

	Mat tmpT1_0, tmpT1_1, tmpT1_2, tmpT1_All = Mat(s1.rows - t1.rows + 1, s1.cols - t1.cols + 1, CV_32F);
	Mat tmpT2_0, tmpT2_1, tmpT2_2, tmpT2_All = Mat(s1.rows - t2.rows + 1, s1.cols - t2.cols + 1, CV_32F);
	Mat tmpT3 = Mat(s2.rows - t3.rows + 1, s2.cols - t3.cols + 1, CV_32F);
	Mat tmpAll = Mat(s1.rows - patchN, s1.cols - patchN, CV_32F);


	matchTemplate(s1C.at(0), t1C.at(0), tmpT1_0, CV_TM_SQDIFF);
	matchTemplate(s1C.at(1), t1C.at(1), tmpT1_1, CV_TM_SQDIFF);
	matchTemplate(s1C.at(2), t1C.at(2), tmpT1_2, CV_TM_SQDIFF);

	sqrt(tmpT1_0, tmpT1_0);
	sqrt(tmpT1_1, tmpT1_1);
	sqrt(tmpT1_2, tmpT1_2);

	add(tmpT1_0, tmpT1_1, tmpT1_All);
	add(tmpT1_2, tmpT1_All, tmpT1_All);

	matchTemplate(s1C.at(0), t2C.at(0), tmpT2_0, CV_TM_SQDIFF);
	matchTemplate(s1C.at(1), t2C.at(1), tmpT2_1, CV_TM_SQDIFF);
	matchTemplate(s1C.at(2), t2C.at(2), tmpT2_2, CV_TM_SQDIFF);

	sqrt(tmpT2_0, tmpT2_0);
	sqrt(tmpT2_1, tmpT2_1);
	sqrt(tmpT2_2, tmpT2_2);

	add(tmpT2_0, tmpT2_1, tmpT2_All);
	add(tmpT2_2, tmpT2_All, tmpT2_All);


	for (int ox = 0; ox < tmpAll.rows; ox++)
	{
		for (int oy = 0; oy < tmpAll.cols; oy++)
		{
			tmpAll.at<float>(ox, oy) = tmpT1_All.at<float>(ox, oy) + tmpT2_All.at<float>(ox, oy);
		}
	}

	tmpAll = tmpAll*0.3333333*a;

	matchTemplate(s2, t3, tmpT3, CV_TM_SQDIFF);
	sqrt(tmpT3, tmpT3);
	tmpT3 = tmpT3*(1 - a);

	for (int ox = 0; ox < tmpAll.rows; ox++)
	{
		for (int oy = 0; oy < tmpAll.cols; oy++)
		{
			tmpAll.at<float>(ox, oy) = tmpT3.at<float>(ox, oy) + tmpAll.at<float>(ox, oy);
		}
	}

	minMaxLoc(tmpAll, &myMin, 0, &minLoc, 0);

	// error tolerance
	for (int ox = 0; ox < tmpAll.rows; ox++)
	{
		for (int oy = 0; oy < tmpAll.cols; oy++)
		{
			if (tmpAll.at<float>(ox, oy) <= et*myMin)
			{
				matchingIndex.push_back({ ox, oy });
			}
		}
	}


	srand(time(NULL));

	p = matchingIndex[rand() % matchingIndex.size()];

	cout << "Double(LAndBSSD):" << p.first << "," << p.second << ":" << myMin << endl;

	return p;
}