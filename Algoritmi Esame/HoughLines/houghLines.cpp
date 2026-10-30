#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int cth1 = 90;  // soglie di Canny
const int cth2 = 160;
const int hth = 90;   // voti minimi per accettare una retta

void houghLines(const Mat& src, Mat& dst) {
    // 1) Smoothing + bordi con Canny
    Mat gauss, edges;
    GaussianBlur(src, gauss, Size(5, 5), 0, 0);
    Canny(gauss, edges, cth1, cth2);

    // 2) Accumulatore: righe = rho in [-maxDist, maxDist], colonne = theta in [0, 180)
    int maxDist = hypot(src.rows, src.cols);
    Mat votes = Mat::zeros(2 * maxDist + 1, 180, CV_32S);

    // 3) Votazione: ogni punto di bordo vota per tutte le rette che ci passano
    for (int y = 0; y < edges.rows; y++)
        for (int x = 0; x < edges.cols; x++)
            if (edges.at<uchar>(y, x) == 255)
                for (int t = 0; t < 180; t++) {
                    double theta = t * CV_PI / 180;
                    int rho = cvRound(x * cos(theta) + y * sin(theta)) + maxDist;
                    votes.at<int>(rho, t)++;
                }

    // 4) Disegno le rette con abbastanza voti (da polari a due punti cartesiani)
    dst = src.clone();
    for (int r = 0; r < votes.rows; r++)
        for (int t = 0; t < votes.cols; t++)
            if (votes.at<int>(r, t) >= hth) {
                double rho = r - maxDist;
                double theta = t * CV_PI / 180;
                double x0 = rho * cos(theta), y0 = rho * sin(theta);  // punto della retta più vicino all'origine
                Point p1(cvRound(x0 - maxDist * sin(theta)), cvRound(y0 + maxDist * cos(theta)));
                Point p2(cvRound(x0 + maxDist * sin(theta)), cvRound(y0 - maxDist * cos(theta)));
                line(dst, p1, p2, Scalar(127), 2);
            }
}

int main(int argc, char** argv) {
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat dst;
    houghLines(src, dst);

    imshow("src", src);
    imshow("hough lines", dst);
    waitKey(0);
    return 0;
}
