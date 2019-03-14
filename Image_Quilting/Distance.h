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

pair<int, int> Distance::BSSD(Mat s, Mat t1, Mat t2, int patchN)
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

pair<int, int> Distance::LSSD(Mat s, Mat t, int patchN)
{
	float myMin = INT_MAX;
	pair<int, int> p;
	vector<pair<int, int>> index, matchingIndex;
	vector<float> ds;

	for (int ox = 0; ox <= s.rows - patchN; ox++)
	{
		for (int oy = 0; oy <= s.cols - patchN; oy++)
		{
			float d = 0;

			for (int ix = 0; ix < t.rows; ix++)
			{
				for (int iy = 0; iy < t.cols; iy++)
				{
					d += sqrt(pow((float)s.at<uchar>(ox + ix, oy + iy) - (float)t.at<uchar>(ix, iy), 2));
				}
			}
			ds.push_back(d);
			index.push_back({ ox, oy });

			if (d < myMin)
			{
				myMin = d;
				p = { ox, oy };
			}
		}
	}

	// error tolerance
	for (int i = 0; i < ds.size(); i++)
	{
		if (ds[i] <= et*myMin)
		{
			matchingIndex.push_back(index[i]);
		}
	}

	srand(time(NULL));

	p = matchingIndex[rand() % matchingIndex.size()];

	cout << "Single(L):" << p.first << "," << p.second << ":" << myMin << endl;

	return p;
}

pair<int, int> Distance::LAndSSD(Mat s1, Mat t1, Mat s2, Mat t2, int patchN, float a)
{
	float myMin = INT_MAX;
	pair<int, int> p;
	vector<pair<int, int>> index, matchingIndex;
	vector<float> ds;

	for (int ox = 0; ox <= s1.rows - patchN; ox++)
	{
		for (int oy = 0; oy <= s1.cols - patchN; oy++)
		{
			float d = 0, d1 = 0, d2 = 0;

			for (int ix = 0; ix < t1.rows; ix++)
			{
				for (int iy = 0; iy < t1.cols; iy++)
				{
					d1 += sqrt(pow((float)s1.at<Vec3b>(ox + ix, oy + iy)[0] - (float)t1.at<Vec3b>(ix, iy)[0], 2)
						+ pow((float)s1.at<Vec3b>(ox + ix, oy + iy)[1] - (float)t1.at<Vec3b>(ix, iy)[1], 2)
						+ pow((float)s1.at<Vec3b>(ox + ix, oy + iy)[2] - (float)t1.at<Vec3b>(ix, iy)[2], 2)) / 3;
				}
			}

			for (int ix = 0; ix < t2.rows; ix++)
			{
				for (int iy = 0; iy < t2.cols; iy++)
				{
					d2 += sqrt(pow((float)s2.at<uchar>(ox + ix, oy + iy) - (float)t2.at<uchar>(ix, iy), 2));
				}
			}

			d = a*d1 + (1 - a)*d2;

			ds.push_back(d);
			index.push_back({ ox, oy });

			if (d < myMin)
			{
				myMin = d;
				p = { ox, oy };
			}
		}
	}

	// error tolerance
	for (int i = 0; i < ds.size(); i++)
	{
		if (ds[i] <= et*myMin)
		{
			matchingIndex.push_back(index[i]);
		}
	}

	srand(time(NULL));

	p = matchingIndex[rand() % matchingIndex.size()];

	cout << "Single(LAndSSD):" << p.first << "," << p.second << ":" << myMin << endl;

	return p;
}

pair<int, int> Distance::LAndBSSD(Mat s1, Mat t1, Mat t2, Mat s2, Mat t3, int patchN, float a)
{
	float myMin = INT_MAX;
	pair<int, int> p;
	vector<pair<int, int>> index, matchingIndex;
	vector<float> ds;

	for (int ox = 0; ox <= s1.rows - patchN; ox++)
	{
		for (int oy = 0; oy <= s1.cols - patchN; oy++)
		{
			float d = 0, d1 = 0, d2 = 0;

			for (int ix = 0; ix < t1.rows; ix++)
			{
				for (int iy = 0; iy < t1.cols; iy++)
				{
					d1 += sqrt(pow((float)s1.at<Vec3b>(ox + ix, oy + iy)[0] - (float)t1.at<Vec3b>(ix, iy)[0], 2)
						+ pow((float)s1.at<Vec3b>(ox + ix, oy + iy)[1] - (float)t1.at<Vec3b>(ix, iy)[1], 2)
						+ pow((float)s1.at<Vec3b>(ox + ix, oy + iy)[2] - (float)t1.at<Vec3b>(ix, iy)[2], 2)) / 3;
				}
			}

			for (int ix = 0; ix < t2.rows; ix++)
			{
				for (int iy = 0; iy < t2.cols; iy++)
				{
					d1 += sqrt(pow((float)s1.at<Vec3b>(ox + ix, oy + iy)[0] - (float)t2.at<Vec3b>(ix, iy)[0], 2)
						+ pow((float)s1.at<Vec3b>(ox + ix, oy + iy)[1] - (float)t2.at<Vec3b>(ix, iy)[1], 2)
						+ pow((float)s1.at<Vec3b>(ox + ix, oy + iy)[2] - (float)t2.at<Vec3b>(ix, iy)[2], 2)) / 3;
				}
			}

			for (int ix = 0; ix < t3.rows; ix++)
			{
				for (int iy = 0; iy < t3.cols; iy++)
				{
					d2 += sqrt(pow((float)s2.at<uchar>(ox + ix, oy + iy) - (float)t3.at<uchar>(ix, iy), 2));
				}
			}

			d = a*d1 + (1 - a)*d2;

			ds.push_back(d);
			index.push_back({ ox, oy });

			if (d < myMin)
			{
				myMin = d;
				p = { ox, oy };
			}
		}
	}

	// error tolerance
	for (int i = 0; i < ds.size(); i++)
	{
		if (ds[i] <= et*myMin)
		{
			matchingIndex.push_back(index[i]);
		}
	}

	srand(time(NULL));

	p = matchingIndex[rand() % matchingIndex.size()];

	cout << "Double(LAndBSSD):" << p.first << "," << p.second << ":" << myMin << endl;

	return p;
}