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

#ifndef LVT_IMAGE_FEATURES_STRUCT_H__
#define LVT_IMAGE_FEATURES_STRUCT_H__

#include "lvt_pose.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/xfeatures2d.hpp>
#include "feature_filter.h"

/*
 * A container for computed image features and their descriptors.
 */

typedef struct track_mp_st {
    int x_idx;
    int y_idx;
    cv::Point2f proj;
    cv::Mat des;
    int track_radius;
    std::vector<std::pair<int, int>> hash_vec;
};

class lvt_image_features_struct
{
  public:
    lvt_image_features_struct();
    void init(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descs,
              int tracking_radius, int hashing_cell_size, int vertical_search_radius, float triangulation_ratio_th,
              float tracking_ratio_th, float desc_dist_th, std::vector<float> *kps_depth = nullptr);

    int find_match_index(const lvt_vector2 &pt, const cv::Mat &desc, float *d1, float *d2) const; // find the best feature in the struct that matches the passed one and return its index, or -1 otherwise. (ratio is the ratio to use for the ratio test).
    int find_match_index(const lvt_vector2& pt, const cv::Mat& desc, const int& tracking_radius) const;
    int th_track_index(track_mp_st track_st) const;
    int row_match(const cv::Point2f &pt, const cv::Mat &desc) const;

    inline cv::Mat get_descriptor(const int index) const { return m_descriptors.row(index); }
    inline cv::Mat get_descriptors() const { return m_descriptors; }  //add for test
    inline const cv::KeyPoint &get_keypoint(const int index) const { return m_keypoints[index]; }
    inline std::vector<cv::KeyPoint> get_keypoints() const { return m_keypoints; }  //add for test
    inline void init_keypoints(std::vector<cv::KeyPoint> kps) { m_keypoints = kps; }  //add only for right camera
    inline void init_descriptors(cv::Mat des) { m_descriptors = des; reset_matched_marks();}  //add only for right camera

    inline int get_features_count() const { return (int)m_keypoints.size(); }

    void mark_as_matched(const int idx, const bool val) { m_matched_marks[idx] = val; }
    bool is_matched(const int idx) const { return m_matched_marks[idx]; }
    void reset_matched_marks() { m_matched_marks = std::vector<bool>(m_keypoints.size(), false); }

    void set_tracking_radius(int tracking_radius) { m_tracking_radius = tracking_radius; }
    int get_tracking_radius() const { return m_tracking_radius; }

    bool is_depth_associated() const { return !m_kps_depths.empty(); }
    float get_keypoint_depth(const int idx) const { return m_kps_depths[idx]; }

    cv::Ptr<cv::DescriptorMatcher>& getMatcher() { return m_matcher; }    //add for test

    std::vector<bool> get_matched_marks() { return m_matched_marks; } //ME!
    std::vector<cv::DMatch> row_match(const cv::Mat& imgL, const cv::Mat& imgR, const cv::Mat& descL, const std::vector<cv::KeyPoint>& kpsL,const std::vector<bool>& marked)const;  //ME!
    
    //std::shared_ptr<Quad_Tree> qtree = nullptr;    //四叉树特征管理器

  private:
    std::vector<cv::KeyPoint> m_keypoints;
    cv::Mat m_descriptors;
    cv::Ptr<cv::DescriptorMatcher> m_matcher;
    std::vector<bool> m_matched_marks;
    int m_cell_size;
    int m_cell_count_x, m_cell_count_y;
    int m_cell_search_radius;
    int m_tracking_radius;
    int m_img_rows, m_img_cols;
    int m_vertical_search_radius;
    float m_triangulation_ratio_th;
    float m_tracking_ratio_th;
    float m_desc_dist_th;
    std::vector<float> m_kps_depths;

    typedef std::vector<int> index_list_t;
    std::vector<std::vector<index_list_t>> m_index_hashmap;    //结构是三维的，[x,y,i]，x，y是二维栅格的索引，i是当前所有2D特征的索引

    typedef std::pair<int, int> index_pair_t;
    inline index_pair_t compute_hashed_index(const cv::Point2f &val, const float cell_size) const
    {
        return index_pair_t(floor(val.y / cell_size), floor(val.x / cell_size));
    }
};

#endif //LVT_IMAGE_FEATURES_STRUCT_H__
