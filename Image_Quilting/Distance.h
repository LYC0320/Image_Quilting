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

private:

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