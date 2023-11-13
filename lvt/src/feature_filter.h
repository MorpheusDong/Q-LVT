#ifndef _FEATURE_FILTER_H_
#define _FEATURE_FILTER_H_

#include <vector>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc.hpp>
#include <omp.h>


typedef struct Point_2D {
	int x;
	int y;
}Point;

#if 0
enum class Region {
	LEFT_UP = 0,
	LEFT_BOTTOM,
	RIGHT_UP,
	RIGHT_BOTTOM,
	OUT_OF_RANGE
};
#endif

typedef struct Quad_Tree_Node {
	Point start_xy;
	int width;
	int height;
	int capacity;
	bool is_leaf;
	std::vector<int> indexes;
	std::vector<cv::Point2f> points;
	Quad_Tree_Node* child[4];
	int depth;
}Quad_Node;

class Quad_Tree {
private:
	Quad_Node* root_;
	int capacity_;
	int max_depth_;
	Quad_Node* create_node_(const Point& start, const int& w, const int& h, const int& d);
	int split_node_(Quad_Node* node);
	int split_to_depth(Quad_Node* node, const int& max_depth);
	int insert_node_(Quad_Node* node, const cv::Point2f& kp, const int& index);
	bool insert_node_r_(Quad_Node* node, const cv::Point2f& kp, const int& index);
	int query_node_(Quad_Node* node, const cv::Point2f& kp);
	void get_near_points(Quad_Node* node, const cv::Point2f& kp, std::vector<int>* kps_indexes);
	//Region find_region(Quad_Node* node, const Point& point);
	void release_node(Quad_Node* node, const int& depth);
	void clear_points(Quad_Node* node);
	void get_near_r(Quad_Node* node, const cv::Point2f& kp, std::vector<int>* kps_indexes, const int r);
	bool filter_good_points(Quad_Node* node, const cv::Point2f& kp, const cv::Mat& pt_des, std::vector<cv::Point2f>* filtered);
	void get_node_points(Quad_Node* node, std::vector<cv::Point2f>* filtered);

public:
	Quad_Tree(const int& start_x, const int& start_y, const int& witdh, const int& height, const int& c);
	~Quad_Tree();
	int insert_(const cv::Point2f& kp, const int& index);
	bool insert_r(const cv::Point2f& kp, const int& index);
	void get_hit_points(const cv::Point2f& kp, std::vector<int>* kps_indexes);
	void get_near_r_points(const cv::Point2f& kp, std::vector<int>* kps_indexes, const int r);
	void release(const int& depth);
	void clear();
	void init_split(const int& depth);
	void set_max_depth(const int& max_d) { max_depth_ = max_d; };
	void adjust_capacity(const int& c) { capacity_ = c; };
	int query_exist_point(const cv::Point2f& kp);
	void get_kept_kps(std::vector<cv::Point2f>* filtered);

};

bool cmp_res(const cv::KeyPoint& p1, const cv::KeyPoint& p2);
int hamming_distance(const cv::Mat& des_l, const cv::Mat& des_r);
bool comp(const std::pair<int, int>& a, const std::pair<int, int>& b);

#endif // !_FEATURE_FILTER_H_