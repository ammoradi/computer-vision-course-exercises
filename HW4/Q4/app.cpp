#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int ac, char** av) 
{
    Mat m,mc;

    mc=imread("line1.bmp",cv::IMREAD_UNCHANGED); /* or line2.bmp */
    cvtColor(mc, m, CV_BGR2GRAY);
    vector<cv::Point2d> p;
    Mat line_fit;
    for (int i=0;i<m.rows;i++)
        for (int j=0;j<m.cols;j++)
            if (m.at<unsigned char>(i,j)!=0)
                p.push_back(Point2d(j,i));
    cout<<"Coordinates : \n";
    cout<<p;
    fitLine(p, line_fit, CV_DIST_L2, 0, 0.01, 0.01);
    double a,b,c;
    b = -line_fit.at<float>(0, 0);
    a = line_fit.at<float>(1, 0);
    c = -(a*line_fit.at<float>(2, 0) + b*line_fit.at<float>(3, 0));
    line(mc,Point2d(0,-c/b),Point2d(m.cols,(-a*m.cols-c)/b),Scalar(0,255,0),2);

    imshow("Fit", mc);
    waitKey();
}