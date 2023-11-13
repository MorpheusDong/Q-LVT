/*********************************************************************
 * BSD 3-Clause License
 *
 * Copyright (c) 2018, Rawashdeh Research Group
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************/
// Author: Mohamed Aladem

#include "lvt_image_features_handler.h"
#include "lvt_logging_utils.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <limits>
#include <thread>

// This function is from code in this answer: http://answers.opencv.org/question/93317/orb-keypoints-distribution-over-an-image/
static void _adaptive_non_maximal_suppresion(std::vector<cv::KeyPoint> &keypoints, const int num_to_keep,
                                             const float tx, const float ty)
{
    // Sort by response
    std::sort(keypoints.begin(), keypoints.end(),
              [&keypoints](const cv::KeyPoint &lhs, const cv::KeyPoint &rhs) {
                  return lhs.response > rhs.response;
              });

    std::vector<cv::KeyPoint> anmsPts;
    anmsPts.reserve(num_to_keep);

    std::vector<float> radii;
    radii.resize(keypoints.size());
    std::vector<float> radiiSorted;
    radiiSorted.resize(keypoints.size());

    const float robustCoeff = 1.11;
    for (int i = 0, count_i = keypoints.size(); i < count_i; i++)
    {
        const float response = keypoints[i].response * robustCoeff;
        float radius = (std::numeric_limits<float>::max)();
        for (int j = 0; j < i && keypoints[j].response > response; j++)
        {
            const cv::Point2f diff_pt = keypoints[i].pt - keypoints[j].pt;
            radius = (std::min)(radius, diff_pt.x * diff_pt.x + diff_pt.y * diff_pt.y);
        }
        radius = sqrtf(radius);
        radii[i] = radius;
        radiiSorted[i] = radius;
    }

    std::sort(radiiSorted.begin(), radiiSorted.end(),
              [&radiiSorted](const float &lhs, const float &rhs) {
                  return lhs > rhs;
              });

    const float decisionRadius = radiiSorted[num_to_keep];
    for (int i = 0, count = radii.size(); i < count; i++)
    {
        if (radii[i] >= decisionRadius)
        {
            keypoints[i].pt.x += tx;
            keypoints[i].pt.y += ty;
            anmsPts.push_back(keypoints[i]);
        }
    }

    anmsPts.swap(keypoints);
}

static void AutoGammaCorrection(const cv::Mat& src, cv::Mat& dst)
{
    const int channels = src.channels();
    const int type = src.type();
    assert(type == CV_8UC1 || type == CV_8UC3);


    //======计算gamma值========//
    auto mean = cv::mean(src);//求均值
    mean[0] = std::log10(0.5) / std::log10(mean[0] / 255);//gamma = -0.3/log10(X)
    if (3 == channels)
    {
        mean[1] = std::log10(0.5) / std::log10(mean[1] / 255);//gamma = -0.3/log10(X)
        mean[2] = std::log10(0.5) / std::log10(mean[2] / 255);//gamma = -0.3/log10(X)

        //多通道图像,对求得的gamm再次平均,避免偏色现象
        auto mean3 = (mean[0] + mean[1] + mean[2]) / 3;
        mean[0] = mean[1] = mean[2] = mean3;
    }

    //=======计算gamma查找表,减少计算量=======//
    //查找表，数组的下标对应图片里面的灰度值
    //lut(0,10)=(50,60,70)表示通道1灰度值为10的像素其对应的值为50;
    // 通道2灰度值为10的像素其对应的值为60;
    // 通道3灰度值为10的像素其对应的值为70
    cv::Mat lut(1, 256, src.type());
    if (1 == channels)
    {
        for (int i = 0; i < 256; ++i)//灰度等级[0,255]
        {
            //将灰度值归一化0-1之间
            float Y = i * 1.0f / 255;// or Y=i*0.00392;
            //求该灰度值gamma校正后的值
            Y = std::pow(Y, mean[0]);

            lut.at<unsigned char>(0, i) = cv::saturate_cast<unsigned char>(Y * 255);
        }
    }
    else if (3 == channels)
    {
        for (int i = 0; i < 256; ++i)//灰度等级[0,255]
        {
            //将灰度值归一化0-1之间
            float Y = i * 1.0f / 255;// or Y=i*0.00392;
            //求该灰度值gamma校正后的值
            auto B = cv::saturate_cast<unsigned char>(std::pow(Y, mean[0]) * 255);
            auto G = cv::saturate_cast<unsigned char>(std::pow(Y, mean[1]) * 255);
            auto R = cv::saturate_cast<unsigned char>(std::pow(Y, mean[2]) * 255);

            lut.at<cv::Vec3b>(0, i) = cv::Vec3b(B, G, R);
        }
    }


    //=========利用查找表进行校正==========//
    cv::LUT(src, lut, dst);
}

lvt_image_features_handler::lvt_image_features_handler(const lvt_parameters &vo_params)
    : m_vo_params(vo_params)
{
    assert(m_vo_params.img_height > 0);    //376
    assert(m_vo_params.img_width > 0);    //1241
    assert(m_vo_params.detection_cell_size > 0);
    assert(m_vo_params.max_keypoints_per_cell > 0);
    assert(m_vo_params.tracking_radius > 0);
    assert(m_vo_params.agast_threshold > 0);

    //将图像栅格化为一个个块
    int num_cells_y = 1 + ((m_vo_params.img_height - 1) / m_vo_params.detection_cell_size); //=2
    int num_cells_x = 1 + ((m_vo_params.img_width - 1) / m_vo_params.detection_cell_size);  //=5
    int s = m_vo_params.detection_cell_size;
    for (int i = 0; i < num_cells_y; i++)
    {
        for (int k = 0; k < num_cells_x; k++)
        {
            int sy = s;
            if ((i == num_cells_y - 1) && ((i + 1) * s > m_vo_params.img_height))
            {
                sy = m_vo_params.img_height - (i * s);
            }
            int sx = s;
            if ((k == num_cells_x - 1) && ((k + 1) * s > m_vo_params.img_width))
            {
                sx = m_vo_params.img_width - (k * s);
            }
            m_sub_imgs_rects.push_back(cv::Rect(k * s, i * s, sx, sy));
        }
    }

    //成员m_th_data存储了特征检测需要的所有对象，如Mat，特征提取器，系统参数等。
    //初始化，左右图像各一个
    m_th_data[0].detector = cv::AgastFeatureDetector::create(m_vo_params.agast_threshold);    //AGAST特征点提取器
    //m_th_data[0].detector = cv::FastFeatureDetector::create(m_vo_params.agast_threshold);
#if TEST_MINE
    m_th_data[0].extractor = cv::xfeatures2d::BEBLID::create(5.00f, cv::xfeatures2d::BEBLID::SIZE_512_BITS);
#endif
    //m_th_data[0].detector = cv::ORB::create(3000);
    //m_th_data[0].extractor = cv::xfeatures2d::FREAK::create();
#if TEST_LVT
    m_th_data[0].extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
#endif
    m_th_data[0].sub_imgs_rects = m_sub_imgs_rects;                                           //栅格图像块
    m_th_data[0].vo_params = &m_vo_params;                                                    //系统参数
    //m_th_data[0].detector = cv::FastFeatureDetector::create(m_vo_params.agast_threshold);
    //m_th_data[0].extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();    

    //m_th_data[1].detector = cv::ORB::create(3000);
    //m_th_data[1].detector = cv::FastFeatureDetector::create(m_vo_params.agast_threshold);
    //m_th_data[1].extractor = cv::xfeatures2d::FREAK::create();
    m_th_data[1].detector = cv::AgastFeatureDetector::create(m_vo_params.agast_threshold);
#if TEST_LVT
    m_th_data[1].extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
#endif
#if TEST_MINE
    m_th_data[1].extractor = cv::xfeatures2d::BEBLID::create(5.00f, cv::xfeatures2d::BEBLID::SIZE_512_BITS);
#endif
    m_th_data[1].sub_imgs_rects = m_sub_imgs_rects;
    m_th_data[1].vo_params = &m_vo_params;

#if TEST_MINE
    //四叉树管理特征
    m_th_data[0].flag = 0;
    m_th_data[1].flag = 1;
    feature_st[0].qtree = std::make_shared<Quad_Tree>(0, 0, m_vo_params.img_width, m_vo_params.img_height, 2);
    feature_st[1].qtree = std::make_shared<Quad_Tree>(0, 0, m_vo_params.img_width, m_vo_params.img_height, 8);
    //feature_st[0].qtree = new Quad_Tree(0, 0, m_vo_params.img_width, m_vo_params.img_height, 8);
    //feature_st[1].qtree = new Quad_Tree(0, 0, m_vo_params.img_width, m_vo_params.img_height, 8);
    feature_st[0].qtree->init_split(3);
    feature_st[0].qtree->set_max_depth(5);
    feature_st[1].qtree->init_split(3);
    feature_st[1].qtree->set_max_depth(5);
   
    is_first_frame = true;
#endif
}

lvt_image_features_handler::~lvt_image_features_handler()
{
}

static void perform_detect_corners(compute_features_data *p, std::vector<cv::KeyPoint> *all_keypoints)
{
    //对每个栅格图块提取特征点
    for (int r = 0; r < p->sub_imgs_rects.size(); r++)
    {
        cv::Rect rect = p->sub_imgs_rects[r];
        cv::Mat sub_img = p->img(rect);
        std::vector<cv::KeyPoint> keypoints;
        keypoints.reserve(p->vo_params->max_keypoints_per_cell);
        p->detector->detect(sub_img, keypoints);
        if (keypoints.size() > p->vo_params->max_keypoints_per_cell)    //特征点较多，做非极大值抑制
        {
            _adaptive_non_maximal_suppresion(keypoints, p->vo_params->max_keypoints_per_cell, (float)rect.x, (float)rect.y);
        }
        else
        {
            for (int i = 0; i < keypoints.size(); i++)
            {
                keypoints[i].pt.x += (float)rect.x;
                keypoints[i].pt.y += (float)rect.y;
            }
        }
        all_keypoints->insert(all_keypoints->end(), keypoints.begin(), keypoints.end());
    }
}


void lvt_image_features_handler::detect_and_filter(compute_features_data* p,std::vector<cv::KeyPoint>* out_kps)
{
    feature_struct* feature_pt = &feature_st[0];

    if (!is_first_frame) {
        feature_pt->qtree->release(3);
    }
    int index = 0;
    //cv::Mat down_img;
    //cv::pyrDown(p->img, down_img);
    std::vector<cv::KeyPoint> detected;
    //std::vector<cv::KeyPoint> detected_s;
    p->detector->detect(p->img, detected);
    //p->detector->detect(down_img, detected_s);
    sort(detected.begin(), detected.end(), cmp_res);
    //sort(detected_s.begin(), detected_s.end(), cmp_res);
    for (int i = 0, count = detected.size(); i < count; ++i) {
        if (feature_pt->qtree->insert_(detected[i].pt, index) != -1) {
            out_kps->push_back(detected[i]);
            index++;
        }
    }

#if 0
    //feature_pt->qtree->adjust_capacity(8);
    for (int i = 0, count = detected_s.size(); i < count; ++i) {
        cv::KeyPoint kp = detected_s[i];
        kp.pt.x = static_cast <int>(2 * kp.pt.x);
        kp.pt.y = static_cast <int>(2 * kp.pt.y);
        if (feature_pt->qtree->insert_r(kp, index)) {
            out_kps->push_back(kp);
            index++;
        }
    }
#endif
    //feature_pt->keypoints = kps_current;
}

//此函数会跑在多线程
void lvt_image_features_handler::perform_compute_features(compute_features_data *p)
{
    std::vector<cv::KeyPoint> all_keypoints;
    cv::Mat desc;
#if TEST_MINE
    if (p->flag == 0) {
        detect_and_filter(p, &all_keypoints);
    }
    else {
        //p->detector->detect(p->img, all_keypoints);
    }
   
#endif
#if TEST_LVT
    all_keypoints.reserve(p->sub_imgs_rects.size() * p->vo_params->max_keypoints_per_cell);
    perform_detect_corners(p, &all_keypoints);    //对所有栅格检测特征点
    if (all_keypoints.size() < LVT_CORNERS_LOW_TH)    //特殊情况，特征点数量不足200
    {
        std::cout << "low features!" << all_keypoints.size() << "\n";
        //system("pause");
        all_keypoints.clear();
        int original_agast_th = p->detector->getThreshold();
        int lowered_agast_th = (double)original_agast_th * 0.5 + 0.5;
        p->detector->setThreshold(lowered_agast_th);    //降低阈值，重新检测一次所有特征点
        perform_detect_corners(p, &all_keypoints);
        p->detector->setThreshold(original_agast_th);
    }
    p->extractor->compute(p->img, all_keypoints, desc);    //计算描述子，[原始图像，特征点，->描述子矩阵]
#endif

#if TEST_MINE
    if (p->flag == 0) {
        p->extractor->compute(p->img, all_keypoints, desc);
        p->features_struct->init(p->img, all_keypoints, desc, p->vo_params->tracking_radius, LVT_HASHING_CELL_SIZE,
            LVT_ROW_MATCHING_VERTICAL_SEARCH_RADIUS, p->vo_params->triangulation_ratio_test_threshold,
            p->vo_params->tracking_ratio_test_threshold, p->vo_params->descriptor_matching_threshold);
    }
    else {
        //p->features_struct->init_keypoints(all_keypoints);
        //p->features_struct->init_descriptors(desc);
    }
#endif
#if TEST_LVT
    p->features_struct->init(p->img, all_keypoints, desc, p->vo_params->tracking_radius, LVT_HASHING_CELL_SIZE,
        LVT_ROW_MATCHING_VERTICAL_SEARCH_RADIUS, p->vo_params->triangulation_ratio_test_threshold,
        p->vo_params->tracking_ratio_test_threshold, p->vo_params->descriptor_matching_threshold);
#endif
}

void lvt_image_features_handler::perform_compute_descriptors_only(compute_features_data *p)
{
    cv::Mat desc;
    const std::vector<cv::Point2f> &ext_kp = *(p->ext_kp);
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(ext_kp.size());
    for (int i = 0, count = ext_kp.size(); i < count; i++)
    {
        cv::KeyPoint kp;
        kp.pt = ext_kp[i];
        keypoints.push_back(kp);
    }
    p->extractor->compute(p->img, keypoints, desc);
    p->features_struct->init(p->img, keypoints, desc, p->vo_params->tracking_radius, LVT_HASHING_CELL_SIZE,
                             LVT_ROW_MATCHING_VERTICAL_SEARCH_RADIUS, p->vo_params->triangulation_ratio_test_threshold,
                             p->vo_params->tracking_ratio_test_threshold, p->vo_params->descriptor_matching_threshold);
}

void lvt_image_features_handler::compute_features(const cv::Mat &img_left, const cv::Mat &img_right,
                                                  lvt_image_features_struct *out_left, lvt_image_features_struct *out_right)
{
#if TEST_LVT
    // Compute left image features on the main thread while the right one in a parallel thread.
    m_th_data[0].img = img_left;
    m_th_data[0].features_struct = out_left;
    m_th_data[1].img = img_right;
    m_th_data[1].features_struct = out_right;
    std::thread th(&lvt_image_features_handler::perform_compute_features, this, &(m_th_data[1]));    //右图像的特征提取是多线程
    perform_compute_features(&(m_th_data[0]));
    th.join();    //阻塞
#endif
#if TEST_MINE
    //cv::Mat img_L, img_R;
    //AutoGammaCorrection(img_left, img_L);
    //AutoGammaCorrection(img_right, img_R);
    m_th_data[0].img = img_left;
    m_th_data[0].features_struct = out_left;
    m_th_data[1].img = img_right;
    m_th_data[1].features_struct = out_right;
    //std::thread th(&lvt_image_features_handler::perform_compute_features, this, &(m_th_data[1]));    //右图像的特征提取是多线程
    perform_compute_features(&(m_th_data[0]));
    //perform_compute_features(&(m_th_data[1]));
    //th.join();    //阻塞
    is_first_frame = false;
#endif

#if 0
    std::cout << "left kps = " << m_th_data[0].features_struct->get_keypoints().size() << ",";
    std::cout << "right kps = " << m_th_data[1].features_struct->get_keypoints().size() << "\n";
    cv::Mat out_img;
    cv::drawKeypoints(img_left, m_th_data[0].features_struct->get_keypoints(), out_img, cv::Scalar(0, 0, 255));
    cv::namedWindow("keypoints MY", cv::WINDOW_NORMAL);
    cv::resizeWindow("keypoints MY", img_left.cols, img_left.rows);
    cv::imshow("keypoints MY", out_img);
    cv::waitKey(0);
#endif
}

void lvt_image_features_handler::compute_descriptors_only(const cv::Mat &img_left, std::vector<cv::Point2f> &ext_kp_left, lvt_image_features_struct *out_left,
                                                          const cv::Mat &img_right, std::vector<cv::Point2f> &ext_kp_right, lvt_image_features_struct *out_right)
{
    m_th_data[0].img = img_left;
    m_th_data[0].ext_kp = &ext_kp_left;
    m_th_data[0].features_struct = out_left;
    m_th_data[1].img = img_right;
    m_th_data[1].ext_kp = &ext_kp_right;
    m_th_data[1].features_struct = out_right;
    std::thread th(&lvt_image_features_handler::perform_compute_descriptors_only, this, &(m_th_data[1]));
    perform_compute_descriptors_only(&(m_th_data[0]));
    th.join();
    m_th_data[0].img = cv::Mat();
    m_th_data[1].img = cv::Mat();
}

void lvt_image_features_handler::compute_features_rgbd(const cv::Mat &img_gray, const cv::Mat &in_img_depth, lvt_image_features_struct *out_struct)
{
    // detect corners in the image as normal
    m_th_data[0].img = img_gray;
    compute_features_data *p = &m_th_data[0];
    std::vector<cv::KeyPoint> all_keypoints;
    all_keypoints.reserve(p->sub_imgs_rects.size() * p->vo_params->max_keypoints_per_cell);
    perform_detect_corners(p, &all_keypoints);
    if (all_keypoints.size() < LVT_CORNERS_LOW_TH)
    {
        all_keypoints.clear();
        int original_agast_th = p->detector->getThreshold();
        int lowered_agast_th = (double)original_agast_th * 0.5 + 0.5;
        p->detector->setThreshold(lowered_agast_th);
        perform_detect_corners(p, &all_keypoints);
        p->detector->setThreshold(original_agast_th);
    }

    // compute descriptors
    cv::Mat desc;
    p->extractor->compute(p->img, all_keypoints, desc);

    // retain corners with valid depth values
    std::vector<float> kps_depths;
    std::vector<cv::KeyPoint> filtered_kps;
    cv::Mat filtered_desc;
    kps_depths.reserve(all_keypoints.size());
    filtered_kps.reserve(all_keypoints.size());
    for (int i = 0; i < all_keypoints.size(); i++)
    {
        const cv::KeyPoint &kp = all_keypoints[i];
        const float d = in_img_depth.at<float>(kp.pt.y, kp.pt.x);
        if (d >= m_vo_params.near_plane_distance && d <= m_vo_params.far_plane_distance)
        {
            kps_depths.push_back(d);
            filtered_kps.push_back(kp);
            filtered_desc.push_back(desc.row(i).clone());
        }
    }

    // Undistort keypoints if the img is distorted
    if (fabs(m_vo_params.k1) > 1e-5)
    {
        cv::Mat kps_mat(filtered_kps.size(), 2, CV_32F);
        for (int i = 0; i < filtered_kps.size(); i++)
        {
            kps_mat.at<float>(i, 0) = filtered_kps[i].pt.x;
            kps_mat.at<float>(i, 1) = filtered_kps[i].pt.y;
        }
        kps_mat = kps_mat.reshape(2);
        cv::Matx33f intrinsics_mtrx(m_vo_params.fx, 0.0, m_vo_params.cx,
                                    0.0, m_vo_params.fy, m_vo_params.cy,
                                    0.0, 0.0, 1.0);
        std::vector<float> dist;
        dist.push_back(m_vo_params.k1);
        dist.push_back(m_vo_params.k2);
        dist.push_back(m_vo_params.p1);
        dist.push_back(m_vo_params.p2);
        dist.push_back(m_vo_params.k3);
        cv::undistortPoints(kps_mat, kps_mat, cv::Mat(intrinsics_mtrx), cv::Mat(dist), cv::Mat(), intrinsics_mtrx);
        kps_mat = kps_mat.reshape(1);
        for (int i = 0; i < filtered_kps.size(); i++)
        {
            cv::KeyPoint &kp = filtered_kps[i];
            kp.pt.x = kps_mat.at<float>(i, 0);
            kp.pt.y = kps_mat.at<float>(i, 1);
        }
    }

    // initialize output structs
    out_struct->init(img_gray, filtered_kps, filtered_desc, m_vo_params.tracking_radius, LVT_HASHING_CELL_SIZE,
                     LVT_ROW_MATCHING_VERTICAL_SEARCH_RADIUS, m_vo_params.triangulation_ratio_test_threshold,
                     m_vo_params.tracking_ratio_test_threshold, m_vo_params.descriptor_matching_threshold, &kps_depths);
}

//匹配左图和右图的所有特征点
void lvt_image_features_handler::row_match(lvt_image_features_struct *features_left, lvt_image_features_struct *features_right,
                                           std::vector<cv::DMatch> *out_matches)
{

    for (int i = 0, count = features_left->get_features_count(); i < count; i++)
    {
        if (features_left->is_matched(i))
        { // if the feature in the left camera image is matched from tracking then ignore it
            continue;
        }
        cv::Mat desc = features_left->get_descriptor(i);
        //注意lvt_image_features_struct里面也有个row_match
        //左图的特征点在右图中找到匹配，方法是在一个矩形范围内+knn匹配，因为假定了双目相机是立体校正的
        //返回右图中匹配到的特征点的索引
        const int match_idx = features_right->row_match(features_left->get_keypoint(i).pt, desc);
        if (match_idx != -1)
        {
            cv::DMatch m;    //匹配关系
            m.queryIdx = i;
            m.trainIdx = match_idx;
            //coarse_matches.push_back(m);
            out_matches->push_back(m);
            features_left->mark_as_matched(i, true);    //标记
            features_right->mark_as_matched(match_idx, true);
        }
    }
}

void lvt_image_features_handler::quad_row_matches(lvt_image_features_struct* features_left, lvt_image_features_struct* features_right, std::vector<cv::DMatch>* out_matches) {

#if 1
    //右目特征点准备
    if (!is_first_frame) {
        feature_st[1].qtree->release(3);
    }

    std::vector<cv::KeyPoint> kps_r;
    kps_r.reserve(2000);
    m_th_data[1].detector->detect(m_th_data[1].img, kps_r);
    sort(kps_r.begin(), kps_r.end(), cmp_res);
    int index = 0;
    std::vector<cv::KeyPoint> kps_r_filtered;
    kps_r_filtered.reserve(1000);
    for (int i = 0, count = kps_r.size(); i < count; ++i) {
        if (feature_st[1].qtree->insert_(kps_r[i].pt, index) != -1) {
            kps_r_filtered.push_back(kps_r[i]);
            index++;
        }
    }

    features_right->init_keypoints(kps_r_filtered);
    cv::Mat des_r;
    m_th_data[1].extractor->compute(m_th_data[1].img, kps_r_filtered, des_r);
    features_right->init_descriptors(des_r);
    //右目特征点准备完成

    for (int i = 0, count = features_left->get_keypoints().size(); i < count; ++i) {
        cv::KeyPoint kp_left = features_left->get_keypoint(i);
        cv::Point2f anchor = features_left->get_keypoint(i).pt;
        std::vector<int> idx_near;
        
        //行搜索
        for (int j = 0; j < 3; ++j) {
            feature_st[1].qtree->get_hit_points(anchor, &idx_near);
            anchor.x -= 38;
            if (anchor.x < 0) {
                break;
            }
        }

        //去重
        std::set<int>s(idx_near.begin(), idx_near.end());
        idx_near.assign(s.begin(), s.end());

        //cv::Mat des_l = features_left->get_descriptor(i);
        //std::vector<std::pair<int, int>> hashvec;

        int min_idx = -1;
        int min_dis1 = 999, min_dis2 = 999;
        int min_idx1 = -1, min_idx2 = -1;

        for (int j = 0,cnt = idx_near.size(); j < cnt; ++j) {
            int near_idx = idx_near[j];
            cv::Point2f cand = kps_r_filtered[near_idx].pt;
            //cv::KeyPoint cand = features_right->get_keypoint(near_idx);
            if (abs(cand.y - kp_left.pt.y) < 3 && !features_right->is_matched(near_idx)) {
                //cv::Mat des_r = features_right->get_descriptor(near_idx);
                cv::Mat des_l = features_left->get_descriptor(i);
                int dis = hamming_distance(des_l, des_r.row(near_idx));
                //hashvec.push_back(std::make_pair(dis, near_idx));
                if (dis < min_dis1) {
                    if (min_idx1 != -1) {
                        min_dis2 = min_dis1;
                        min_idx2 = min_idx1;
                    }
                    min_dis1 = dis;
                    min_idx1 = near_idx;
                }
                else if (dis < min_dis2) {
                    min_dis2 = dis;
                    min_idx2 = near_idx;
                }
                else {
                }
            }
        }

        if (min_idx2 != -1) {
            if (min_dis1 < 0.6 * min_dis2) {
                min_idx = min_idx1;
            }
        }
        else {
            if (min_dis1 < 52) {
                min_idx = min_idx1;
            }
        }

#if 0
        if (hashvec.size() > 1) {
            sort(hashvec.begin(), hashvec.end(), comp);
            auto it_min = hashvec.begin();
            auto it_min2 = hashvec.begin() + 1;
            if (it_min->first < 0.6 * it_min2->first) {    //ratio test
                min_idx = it_min->second;
            }
        }
        else if (hashvec.size() == 1) {
            if (hashvec.begin()->first < 52) {
                min_idx = hashvec.begin()->second;
            }
        }
        else {
        }
#endif

        if (min_idx != -1) {
            cv::DMatch m;    //匹配关系
            m.queryIdx = i;
            m.trainIdx = min_idx;
            out_matches->push_back(m);
            features_left->mark_as_matched(i, true);    //标记
            features_right->mark_as_matched(min_idx, true);
        }
    }

#endif 

#if 0
    std::vector<cv::Point2f> points_left;
    std::vector<cv::Point2f> points_right;
    std::vector<uchar> status;
    std::vector<float> error;
    int points_size = features_left->get_keypoints().size();
    for (int i = 0; i < points_size; ++i) {
        points_left.push_back(features_left->get_keypoint(i).pt);
    }
    int tracked_cnt = 0;
    cv::calcOpticalFlowPyrLK(m_th_data[0].img, m_th_data[1].img, points_left, points_right, status, error, cv::Size(50, 50), 1);
    for (int i = 0; i < points_size; ++i) {
        if (points_right[i].x < 0 || points_right[i].x > 1240 || points_right[i].y < 0 || points_right[i].y > 375) {
            continue;
        }
        //if (features_left->is_matched(i))
        //{ // if the feature in the left camera image is matched from tracking then ignore it
            //continue;
        //}
        //double dist = abs(points_new[i].x - points_old[i].x) + abs(points_new[i].y - points_old[i].y);
        int y_dis = abs(points_right[i].y - points_left[i].y);
        //int x_dis = points_left[i].x - points_right[i].x;
        if (status[i] && y_dis<3) {
            //tracked_cnt++;
            std::vector<int> near_indexs;
            feature_st[1].qtree->get_hit_points(cv::KeyPoint(points_right[i], 1.f), &near_indexs);
            int min_idx = -1;
            std::vector<std::pair<int, int>> hashvec;
            cv::Mat des_l = features_left->get_descriptor(i);
            for (int j = 0, cnt = near_indexs.size(); j < cnt; ++j) {
                if (features_right->is_matched(near_indexs[j])) {
                    continue;
                }
                cv::Mat des_r = features_right->get_descriptor(near_indexs[j]);
                int dis = hamming_distance(des_l, des_r);
                hashvec.push_back(std::make_pair(dis, near_indexs[j]));
            }
            if (hashvec.size() > 1) {
                sort(hashvec.begin(), hashvec.end(), comp);
                auto it_min = hashvec.begin();
                auto it_min2 = hashvec.begin() + 1;
                if (it_min->first < 0.6 * it_min2->first) {    //ratio test
                    min_idx = it_min->second;
                }
            }
            else if (hashvec.size() == 1) {
                if (hashvec.begin()->first < 52) {
                    min_idx = hashvec.begin()->second;
                }
            }
            else {
            }
            if (min_idx != -1) {
                tracked_cnt++;
                cv::DMatch m;    //匹配关系
                m.queryIdx = i;
                m.trainIdx = min_idx;
                out_matches->push_back(m);
                features_left->mark_as_matched(i, true);    //标记
                features_right->mark_as_matched(min_idx, true);

            }
        }
    }
#endif
#if 0
            int match_idx = feature_st[1].qtree->query_exist_point(cv::KeyPoint(points_right[i], 1.f));
            if (match_idx != -1) {
                tracked_cnt++;
                cv::Mat des_l = features_left->get_descriptor(i);
                cv::Mat des_r = features_right->get_descriptor(match_idx);
                int dis = hamming_distance(des_l, des_r);
                if (dis < 52) {
                    tracked_cnt++;
                    cv::DMatch m;    //匹配关系
                    m.queryIdx = i;
                    m.trainIdx = match_idx;
                    out_matches->push_back(m);
                    features_left->mark_as_matched(i, true);    //标记
                    features_right->mark_as_matched(match_idx, true);
                }
            }
        }
    }
#endif

}


