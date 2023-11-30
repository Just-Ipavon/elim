#include <opencv2/opencv.hpp>

void splitAndMerge(const cv::Mat& input, cv::Mat& output, int minRegionSize) {
    if (input.rows * input.cols <= minRegionSize) {
        output = input.clone();
        return;
    }

    cv::Mat blurred;
    cv::blur(input, blurred, cv::Size(3, 3));

    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(blurred, labels, stats, centroids);

    output = cv::Mat::zeros(input.size(), CV_8UC1);

    for (int i = 1; i < numComponents; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= minRegionSize) {
            cv::Mat mask = (labels == i);
            output.setTo(stats.at<int>(i, cv::CC_STAT_AREA), mask);
        }
    }
}

int main() {
    cv::Mat inputImage = cv::imread("input_image.jpg", cv::IMREAD_GRAYSCALE);

    if (inputImage.empty()) {
        std::cerr << "Could not read the image!";
        return -1;
    }

    cv::Mat outputImage;
    int minRegionSize = 1000; // Set your desired minimum region size

    splitAndMerge(inputImage, outputImage, minRegionSize);

    cv::imshow("Original Image", inputImage);
    cv::imshow("Split and Merge Result", outputImage);
    cv::waitKey(0);

    return 0;
}
