#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <iostream>
#include <queue>

using namespace cv;
using namespace std;


// Raw gradient. No denoising
void gradient(const Mat&Ic, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	G2 = Mat(m, n, CV_32F);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
            float ix, iy;
            if (i == 0 || i == m - 1)
                iy = 0;
            else
                iy = (float(I.at<uchar>(i + 1, j)) - float(I.at<uchar>(i - 1, j))) / 2;
            if (j == 0 || j == n - 1)
                ix = 0;
            else
                ix = (float(I.at<uchar>(i, j + 1)) - float(I.at<uchar>(i, j - 1))) / 2;
            G2.at<float>(i, j) = sqrt(ix*ix + iy * iy);
		}
	}
    
}

// Gradient (and derivatives), Sobel denoising
void sobel(const Mat&Ic, Mat& Ix, Mat& Iy, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	Ix = Mat(m, n, CV_32F);
	Iy = Mat(m, n, CV_32F);
	G2 = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
            float ix, iy;
            if (i == 0 || i == m - 1 || j == 0 || j == n - 1){
                iy = 0;
                ix = 0;
            }
            else {
             ix = - float(I.at<uchar>(i+1, j - 1)) + float(I.at<uchar>(i+1, j + 1)) -    2*float(I.at<uchar>(i, j - 1)) + 2*float(I.at<uchar>(i, j + 1)) - float(I.at<uchar>(i-1, j - 1)) + float(I.at<uchar>(i-1, j + 1));
             iy = - float(I.at<uchar>(i+1, j - 1)) - 2*float(I.at<uchar>(i+1, j))  - float(I.at<uchar>(i+1, j + 1)) + float(I.at<uchar>(i-1, j - 1)) + 2*float(I.at<uchar>(i-1, j)) + float(I.at<uchar>(i-1, j + 1));
             Ix.at<float>(i, j) = ix;
             Iy.at<float>(i, j) = iy;
             G2.at<float>(i, j) = sqrt(ix*ix + iy * iy);
            }
		}
	}
}

// Gradient thresholding, default = do not denoise
Mat threshold(const Mat& Ic, float s, bool denoise = false)
{
	Mat Ix, Iy, G2;
    float g;
	if (denoise)
		sobel(Ic, Ix, Iy, G2);
	else
		gradient(Ic, G2);
	int m = Ic.rows, n = Ic.cols;
    //cout << G2 <<endl;
	Mat C(m, n, CV_8U);
	for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++){
            g = G2.at<float>(i, j);
            if (g > s) C.at<uchar>(i, j) = g;
            else C.at<uchar>(i, j) = 0;
        }
	return C;
}

// Canny edge detector
Mat canny(const Mat& Ic, float s1, float s2)
{
	Mat Ix, Iy, G2;
	sobel(Ic, Ix, Iy, G2);

	int m = Ic.rows, n = Ic.cols;
	Mat Max(m, n, CV_8U);	// Max pixels ( max in the direction of the gradient )
	queue<Point> Q;			// Enqueue weak pixels ( Max pixels for which s1 < G2 < s2 )
    Mat C(m, n, CV_8U);     // Strong pixels ( Max pixels for which G2 >= s2 )
    C.setTo(0);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
            float curr_pix = G2.at<float>(i,j);
            float neigh_pix1 = 255, neigh_pix2 = 255;
            
            int dir = int(atan(Iy.at<float>(i, j) / Ix.at<float>(i, j)) / M_PI *180.);
            if (dir < 22.5 and dir>-22.5) {
                neigh_pix1 = G2.at<float>(i,j+1);
                neigh_pix2 = G2.at<float>(i,j-1);
            }
            else if (dir >= 22.5 and dir<67.5) {
                neigh_pix1 = G2.at<float>(i-1,j+1);
                neigh_pix2 = G2.at<float>(i+1,j-1);
            }
            else if ((dir <= 90 and dir>=67.5) or (dir >= -90 and dir<67.5))
            {
                neigh_pix1 = G2.at<float>(i-1,j);
                neigh_pix2 = G2.at<float>(i+1,j);
            }
            else if (dir <= -67.5 and dir>=-22.5)
            {
                neigh_pix1 = G2.at<float>(i+1,j+1);
                neigh_pix2 = G2.at<float>(i-1,j-1);
            }
            
            
            if (neigh_pix1>curr_pix or neigh_pix2>curr_pix)
                Max.at<uchar>(i,j) = 0;
            else{
                Max.at<uchar>(i,j) = curr_pix;
                if (curr_pix>s1 and curr_pix<s2)
                    Q.push(Point(j,i)); // Beware: Mats use row,col, but points use x,y
                else if (curr_pix>=s2)
                    C.at<uchar>(i,j) = curr_pix;
            }
		}
	}

	// Propagate strong pixels
	while (!Q.empty()) {
		int i = Q.front().y, j = Q.front().x;
		Q.pop();
        int sum = 0;
        for (int k = max(i-1,0); k <= min(i+1,m-1); k++) {
            for (int l = max(j-1,0); l <= min(j+1,n-1); l++) {
                sum+=C.at<uchar>(k,l);
                if (sum>0) {
                    C.at<uchar>(i,j) = G2.at<float>(i,j);
                    break;
                }
            }
            if (sum>0) break;
        }
	}
	return C;
}

int main()
{
	Mat I = imread("../images/road.jpg");

	imshow("Input", I);
	imshow("Threshold", threshold(I, 15));
	imshow("Threshold + denoising", threshold(I, 15, true));
	imshow("Canny", canny(I, 50, 100));

	waitKey();

	return 0;
}
