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

#include "lvt_image_features_struct.h"
#include "lvt_logging_utils.h"

lvt_image_features_struct::lvt_image_features_struct() : m_img_rows(0), m_img_cols(0),
m_tracking_radius(0), m_cell_count_x(-1), m_cell_count_y(-1), m_cell_search_radius(0), m_cell_size(25),
m_vertical_search_radius(2), m_triangulation_ratio_th(0.6), m_tracking_ratio_th(0.8), m_desc_dist_th(25.0)
{
}


void lvt_image_features_struct::init(const cv::Mat& in_image, std::vector<cv::KeyPoint>& in_keypoints, const cv::Mat& in_desc,
    int in_tracking_radius, int in_hashing_cell_size, int in_vertical_search_radius,
    float in_triangulation_ratio_th, float in_tracking_ratio_th, float in_desc_dist_th, std::vector<float> *kps_depth)
{
    m_cell_size = in_hashing_cell_size;
    m_vertical_search_radius = in_vertical_search_radius;
    m_triangulation_ratio_th = in_triangulation_ratio_th;
    m_tracking_ratio_th = in_tracking_ratio_th;
    m_desc_dist_th = in_desc_dist_th;
    m_img_rows = in_image.rows;
    m_img_cols = in_image.cols;
    m_tracking_radius = in_tracking_radius;
    const float k_cell_size = static_cast<float>(m_cell_size);
    m_cell_count_x = std::ceil(m_img_cols / k_cell_size);
    m_cell_count_y = std::ceil(m_img_rows / k_cell_size);
    m_matcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher(cv::NORM_HAMMING, false));
    m_keypoints.swap(in_keypoints);
    m_descriptors = in_desc;
    m_cell_search_radius = (m_tracking_radius == m_cell_size) ? 1 : std::ceil((float)m_tracking_radius / k_cell_size);
    m_index_hashmap.resize(m_cell_count_y);
    for (int i = 0; i < m_cell_count_y; i++) {
        m_index_hashmap[i].resize(m_cell_count_x);
    }
    for (int i = 0, count = m_keypoints.size(); i < count; i++) {
        index_pair_t hashed_idx = compute_hashed_index(m_keypoints[i].pt, k_cell_size);
        m_index_hashmap[hashed_idx.first][hashed_idx.second].push_back(i);
    }
    reset_matched_marks();
    if (kps_depth) {
        m_kps_depths = *kps_depth;
    }
}

//寻找投影到像素平面的地图点对应的匹配
int lvt_image_features_struct::find_match_index(const lvt_vector2& sl_pt, const cv::Mat& desc, float *d1, float *d2)const
{
    const cv::Point2f pt(sl_pt.x(), sl_pt.y());    //转为2D坐标
    //std::cout << "(" << pt.x << "," << pt.y << ")\n";
    const index_pair_t hash_idx = compute_hashed_index(pt, (float)m_cell_size);    //计算所处的栅格的索引
    int start_y = hash_idx.first - m_cell_search_radius;    //m_cell_search_radius=1
    if (start_y < 0)
        start_y = 0;
    int end_y = hash_idx.first + m_cell_search_radius + 1;
    if (end_y > m_cell_count_y)
        end_y = m_cell_count_y;
    int start_x = hash_idx.second - m_cell_search_radius;
    if (start_x < 0)
        start_x = 0;
    int end_x = hash_idx.second + m_cell_search_radius + 1;
    if (end_x > m_cell_count_x)
        end_x = m_cell_count_x;

#if 0
    const float r2 = static_cast<float>(m_tracking_radius*m_tracking_radius);
    cv::Mat mask(cv::Mat::zeros(1, m_keypoints.size(), CV_8UC1));
    for (int i = start_y; i < end_y; i++) {
        for (int k = start_x; k < end_x; k++) {
            //m_index_hashmap[i][k]是检测的栅格对应的特征点索引数组
            const index_list_t& kp_index_list = m_index_hashmap[i][k];
            for (size_t kp = 0, count = kp_index_list.size(); kp < count; kp++) {    //遍历该索引数组，找到最佳匹配
                const int kp_idx = kp_index_list[kp];
                if (!m_matched_marks[kp_idx]) {
                    const float dx = m_keypoints[kp_idx].pt.x - pt.x;
                    const float dy = m_keypoints[kp_idx].pt.y - pt.y;
                    if ((dx*dx + dy * dy) < r2) {    //计算检测区域内的特征点，距离在这个半径以内的
                        mask.at<uint8_t>(0, kp_index_list[kp]) = 1;    //它们作为候选点
                    }
                }
            }
        }
    }

    //采用KNN+ratio test
    std::vector< std::vector<cv::DMatch> > matches;
    m_matcher->knnMatch(desc, m_descriptors, matches, 2, mask);    //根据描述符从候选点中找到最佳的
    if (matches[0].size() > 1) {
        float d_ratio = matches[0][0].distance / matches[0][1].distance;
        if (d_ratio < m_tracking_ratio_th) {
            *d1 = matches[0][0].distance;
            *d2 = matches[0][1].distance;
            return matches[0][0].trainIdx;
        }
    } else if ((matches[0].size() == 1) && (matches[0][0].distance <= m_desc_dist_th)) {
        *d1 = matches[0][0].distance;
        *d2 = -1.0;
        return matches[0][0].trainIdx;
    }

    return -1;
#endif
#if 1
    //std::vector<int> near_idxes;
    //int min_idx = -1;
    const float r2 = static_cast<float>(m_tracking_radius * m_tracking_radius);
    //std::vector<std::pair<int, int>> hashvec;

    int min_dis1 = 999, min_dis2 = 999;
    int min_idx1 = -1, min_idx2 = -1;
    for (int i = start_y; i < end_y; i++) {
        for (int k = start_x; k < end_x; k++) {
            const index_list_t& kp_index_list = m_index_hashmap[i][k];
            for (size_t kp = 0, count = kp_index_list.size(); kp < count; kp++) {
                const int kp_idx = kp_index_list[kp];
                if (!m_matched_marks[kp_idx]) {
                    const float dx = m_keypoints[kp_idx].pt.x - pt.x;
                    const float dy = m_keypoints[kp_idx].pt.y - pt.y;
                    if ((dx * dx + dy * dy) < r2) {
                        int dis = hamming_distance(desc, m_descriptors.row(kp_idx));
                        //hashvec.push_back(std::make_pair(dis, kp_idx));
                        if (dis < min_dis1) {
                            if (min_idx1 != -1) {
                                min_dis2 = min_dis1;
                                min_idx2 = min_idx1;
                            }
                            min_dis1 = dis;
                            min_idx1 = kp_idx;
                        }
                        else if (dis < min_dis2) {
                            min_dis2 = dis;
                            min_idx2 = kp_idx;
                        }
                        else {
                        }
                    }
                }
            }
        }
    }

    if (min_idx2 != -1) {
        if (min_dis1 < 0.8 * min_dis2) {
            return min_idx1;
        }
    }
    else {
        if (min_dis1 < 52) {
            return min_idx1;
        }
    }

    return -1;
#endif

# if 0
    if (hashvec.size() > 1) {
        sort(hashvec.begin(), hashvec.end(), comp);
        int dis_min1 = hashvec.begin()->first;
        auto it_min2 = hashvec.begin() + 1;
        int dis_min2 = it_min2->first;
        if (dis_min1 < 0.8 * dis_min2) {    //ratio test
            min_idx = hashvec.begin()->second;
            *d1 = dis_min1;
            *d2 = dis_min2;
        }
    }
    else if (hashvec.size() == 1) {
        if (hashvec.begin()->first < 52) {
            min_idx = hashvec.begin()->second;
            *d1 = hashvec.begin()->first;
        }
    }
    else {
    }
#endif

    //return min_idx;
}

int lvt_image_features_struct::find_match_index(const lvt_vector2& sl_pt, const cv::Mat& desc, const int& tracking_radius) const
{
    const cv::Point2f pt(sl_pt.x(), sl_pt.y());    //转为2D坐标
    const index_pair_t hash_idx = compute_hashed_index(pt, (float)m_cell_size);    //计算所处的栅格的索引
    int start_y = hash_idx.first - m_cell_search_radius;    //m_cell_search_radius=1
    if (start_y < 0)
        start_y = 0;
    int end_y = hash_idx.first + m_cell_search_radius + 1;
    if (end_y > m_cell_count_y)
        end_y = m_cell_count_y;
    int start_x = hash_idx.second - m_cell_search_radius;
    if (start_x < 0)
        start_x = 0;
    int end_x = hash_idx.second + m_cell_search_radius + 1;
    if (end_x > m_cell_count_x)
        end_x = m_cell_count_x;

    float r2 = static_cast<float>(tracking_radius * tracking_radius);
    std::vector<std::pair<int, int>> hashvec;
    for (int i = start_y; i < end_y; i++) {
        for (int k = start_x; k < end_x; k++) {
            const index_list_t& kp_index_list = m_index_hashmap[i][k];
            for (size_t kp = 0, count = kp_index_list.size(); kp < count; kp++) {
                const int kp_idx = kp_index_list[kp];
                if (!m_matched_marks[kp_idx]) {
                    const float dx = m_keypoints[kp_idx].pt.x - pt.x;
                    const float dy = m_keypoints[kp_idx].pt.y - pt.y;
                    if ((dx * dx + dy * dy) < r2) {
                        cv::Mat des = m_descriptors.row(kp_idx);
                        int dis = hamming_distance(desc, des);
                        hashvec.push_back(std::make_pair(dis, kp_idx));
                    }
                }
            }
        }
    }

    int min_idx = -1;
    if (hashvec.size() > 1) {
        sort(hashvec.begin(), hashvec.end(), comp);
        int dis_min1 = hashvec.begin()->first;
        auto it_min2 = hashvec.begin() + 1;
        int dis_min2 = it_min2->first;
        if (dis_min1 < 0.8 * dis_min2) {    //ratio test
            min_idx = hashvec.begin()->second;
        }
    }
    else if (hashvec.size() == 1) {
        if (hashvec.begin()->first < 52) {
            min_idx = hashvec.begin()->second;
        }
    }
    else {
    }



    return min_idx;
}

std::mutex mtx;

int lvt_image_features_struct::th_track_index(track_mp_st track_st) const
{
    float r2 = static_cast<float>(track_st.track_radius);
    const index_list_t& kp_index_list = m_index_hashmap[track_st.y_idx][track_st.x_idx];
    for (size_t kp = 0, count = kp_index_list.size(); kp < count; kp++) {
        const int kp_idx = kp_index_list[kp];
        if (!m_matched_marks[kp_idx]) {
            const float dx = m_keypoints[kp_idx].pt.x - track_st.proj.x;
            const float dy = m_keypoints[kp_idx].pt.y - track_st.proj.y;
            if ((dx * dx + dy * dy) < r2) {
                cv::Mat des = m_descriptors.row(kp_idx);
                int dis = hamming_distance(track_st.des, des);
                mtx.lock();
                track_st.hash_vec.push_back(std::make_pair(dis, kp_idx));
                mtx.unlock();
            }
        }
    }

    return 0;
}

int lvt_image_features_struct::row_match(const cv::Point2f& pt, const cv::Mat& desc)const
{
    //搜索范围
    int start_y = int(pt.y) - m_vertical_search_radius;
    if (start_y < 0)
        start_y = 0;
    int end_y = int(pt.y) + m_vertical_search_radius;
    if (end_y > m_img_rows)
        end_y = m_img_rows;

    cv::Mat mask(cv::Mat::zeros(1, m_keypoints.size(), CV_8UC1));
    for (int i = 0, count = m_keypoints.size(); i < count; i++) {
        const cv::Point2f kp_pt = m_keypoints[i].pt;
        if (!m_matched_marks[i] && kp_pt.y >= start_y && kp_pt.y <= end_y) {
            mask.at<uint8_t>(0, i) = 1;
        }
    }

    //k近邻匹配
    std::vector< std::vector<cv::DMatch> > matches;
    m_matcher->knnMatch(desc, m_descriptors, matches, 2,mask);

    //ratio test
    if (((matches[0].size() > 1) && (matches[0][0].distance / matches[0][1].distance) < m_triangulation_ratio_th) ||
        ((matches[0].size() == 1) && (matches[0][0].distance <= m_desc_dist_th))) {
        return matches[0][0].trainIdx;
    }


    /*else*/
    return -1;
}
