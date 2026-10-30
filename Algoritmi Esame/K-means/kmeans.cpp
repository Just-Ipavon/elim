#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

const int k = 6;          // numero di cluster
const double th = 0.5;    // mi fermo quando i centri si spostano (in media) meno di così

void kmeansSegmentation(const Mat& src, Mat& dst) {
    // 1) Inizializzo i k centri con il colore di pixel casuali
    RNG rng(getTickCount());
    vector<Vec3f> centers(k);
    for (int c = 0; c < k; c++)
        centers[c] = src.at<Vec3b>(rng.uniform(0, src.rows), rng.uniform(0, src.cols));

    Mat labels(src.size(), CV_32S);
    double shift = INFINITY;
    while (shift > th) {
        // 2) Assegnamento: ogni pixel va al centro più vicino (distanza euclidea sui colori)
        for (int y = 0; y < src.rows; y++)
            for (int x = 0; x < src.cols; x++) {
                Vec3f px = src.at<Vec3b>(y, x);
                double minDist = INFINITY;
                for (int c = 0; c < k; c++) {
                    double d = norm(px - centers[c]);
                    if (d < minDist) {
                        minDist = d;
                        labels.at<int>(y, x) = c;
                    }
                }
            }

        // 3) Aggiornamento: ogni centro diventa la media dei suoi pixel
        vector<Vec3f> sum(k, Vec3f(0, 0, 0));
        vector<int> count(k, 0);
        for (int y = 0; y < src.rows; y++)
            for (int x = 0; x < src.cols; x++) {
                int c = labels.at<int>(y, x);
                sum[c] += Vec3f(src.at<Vec3b>(y, x));
                count[c]++;
            }
        shift = 0;
        for (int c = 0; c < k; c++)
            if (count[c] > 0) {  // un cluster vuoto mantiene il suo centro
                Vec3f newCenter = sum[c] / count[c];
                shift += norm(newCenter - centers[c]);
                centers[c] = newCenter;
            }
        shift /= k;
    }

    // 4) Coloro ogni pixel con il colore del suo centro
    dst.create(src.size(), src.type());
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            dst.at<Vec3b>(y, x) = centers[labels.at<int>(y, x)];
}

int main(int argc, char** argv) {
    Mat src = imread(argv[1], IMREAD_COLOR);
    if (src.empty()) return -1;

    Mat dst;
    kmeansSegmentation(src, dst);

    imshow("src", src);
    imshow("kmeans", dst);
    waitKey(0);
    return 0;
}
