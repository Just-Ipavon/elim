#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

const double sth = 10;  // split se la deviazione standard supera questa soglia
const int tsize = 8;    // dimensione minima di una regione
const int mth = 5;      // merge se le medie differiscono meno di questa soglia

/*
 Quadranti in ordine CIRCOLARE: i e (i+1)%4 sono sempre adiacenti
  _______
 | 0 | 1 |
 |---+---|
 | 3 | 2 |
  -------
*/
struct TNode {
    Rect region;
    double mean, stddev;
    vector<TNode*> children = vector<TNode*>(4, nullptr);
    vector<TNode*> merged;                     // nodi uniti in questa regione
    vector<bool> mergedB = vector<bool>(4, false);  // quali figli sono stati uniti
    TNode(Rect R) { region = R; }
};

// 1) SPLIT: divido in 4 finché la regione non è omogenea o troppo piccola
TNode* split(Mat& img, Rect R) {
    TNode* root = new TNode(R);
    Scalar mean, stddev;
    meanStdDev(img(R), mean, stddev);
    root->mean = mean[0];
    root->stddev = stddev[0];

    if (R.width > tsize && root->stddev > sth) {
        int h = R.width / 2;
        root->children[0] = split(img, Rect(R.x,     R.y,     h, h));  // alto sx
        root->children[1] = split(img, Rect(R.x + h, R.y,     h, h));  // alto dx
        root->children[2] = split(img, Rect(R.x + h, R.y + h, h, h));  // basso dx
        root->children[3] = split(img, Rect(R.x,     R.y + h, h, h));  // basso sx
    }
    rectangle(img, R, Scalar(0));  // disegno il quadtree
    return root;
}

bool similar(TNode* a, TNode* b) {
    return abs(a->mean - b->mean) < mth;
}

void addMerged(TNode* root, int i) {
    root->merged.push_back(root->children[i]);
    root->mergedB[i] = true;
}

// 2) MERGE: unisco due figli adiacenti simili (ed eventualmente i loro vicini), il resto lo ricorro
void merge(TNode* root) {
    if (!root->children[0]) {  // foglia: la regione è se stessa
        root->merged.push_back(root);
        return;
    }
    for (int i = 0; i < 4; i++) {
        int next = (i + 1) % 4, opp = (i + 2) % 4, prev = (i + 3) % 4;
        if (similar(root->children[i], root->children[next])) {
            addMerged(root, i);
            addMerged(root, next);
            if (similar(root->children[next], root->children[opp])) addMerged(root, opp);
            if (similar(root->children[i], root->children[prev])) addMerged(root, prev);
            break;
        }
    }
    for (int i = 0; i < 4; i++)
        if (!root->mergedB[i])
            merge(root->children[i]);
}

// 3) SEGMENT: coloro ogni gruppo unito con la media delle medie, poi ricorro sui figli non uniti
void segment(TNode* root, Mat& img) {
    if (!root->merged.empty()) {
        double val = 0;
        for (TNode* node : root->merged)
            val += node->mean;
        val /= root->merged.size();
        for (TNode* node : root->merged)
            img(node->region) = (int)val;
    }
    for (int i = 0; i < 4; i++)
        if (root->children[i] && !root->mergedB[i])
            segment(root->children[i], img);
}

int main(int argc, char** argv) {
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    GaussianBlur(src, src, Size(5, 5), 0, 0);

    // Ritaglio un quadrato di lato potenza di 2
    int s = pow(2, (int)log2(min(src.rows, src.cols)));
    src = src(Rect(0, 0, s, s)).clone();
    Mat seg = src.clone();

    TNode* root = split(src, Rect(0, 0, s, s));
    merge(root);
    segment(root, seg);

    imshow("quad tree", src);
    imshow("segmented", seg);
    waitKey(0);
    return 0;
}
