#include "feature_filter.h"
#include <algorithm>
#include <cmath>

Quad_Tree::Quad_Tree(const int& origin_x, const int& origin_y, const int& witdh, const int& height, const int& c) {
	root_ = nullptr;
	Point origin;
	origin.x = origin_x;
	origin.y = origin_y;
	capacity_ = c;
	max_depth_ = 4;
	root_ = create_node_(origin, witdh, height, 0);
}

Quad_Tree::~Quad_Tree() {
	if (root_) {
		release_node(root_, 0);
		delete root_;
	}
}


Quad_Node* Quad_Tree::create_node_(const Point& start, const int& w, const int& h, const int& d) {
	Quad_Node* new_node = new Quad_Node;
	new_node->start_xy = start;
	new_node->width = w;
	new_node->height = h;
	new_node->capacity = capacity_;
	new_node->is_leaf = true;

	int i = 0;
	new_node->child[i++] = nullptr;
	new_node->child[i++] = nullptr;
	new_node->child[i++] = nullptr;
	new_node->child[i++] = nullptr;
	new_node->depth = d;
	return new_node;
}

int Quad_Tree::split_node_(Quad_Node* node) {
	node->is_leaf = false;

	int sub_w = (node->width + 1) / 2;
	int sub_h = (node->height + 1) / 2;
	int next_depth = node->depth + 1;

	Point lu_start = node->start_xy;
	Point lb_start;
	lb_start.x = node->start_xy.x;
	lb_start.y = node->start_xy.y + sub_h;
	Point ru_start;
	ru_start.x = node->start_xy.x + sub_w;
	ru_start.y = node->start_xy.y;
	Point rb_start;
	rb_start.x = node->start_xy.x + sub_w;
	rb_start.y = node->start_xy.y + sub_h;

	node->child[0] = create_node_(lu_start, sub_w, sub_h, next_depth);
	node->child[1] = create_node_(ru_start, sub_w, sub_h, next_depth);
	node->child[2] = create_node_(lb_start, sub_w, sub_h, next_depth);
	node->child[3] = create_node_(rb_start, sub_w, sub_h, next_depth);

	for (int i = 0, count = node->points.size(); i < count; ++i) {
		int area_idx_x = (node->points[i].x - 1 - lu_start.x) / sub_w;    //向下取整
		int area_idx_y = (node->points[i].y - 1 - lu_start.y) / sub_h;
		int area_idx = 2 * area_idx_y + area_idx_x;
		insert_node_(node->child[area_idx], node->points[i], node->indexes[i]);
	}
	node->points.clear();
	node->indexes.clear();

	return 0;
}

int Quad_Tree::split_to_depth(Quad_Node* node, const int& max_depth)
{
	if (node->depth == max_depth) {
		return 1;
	}
	else {
		node->is_leaf = false;
		//node->depth = depth;
		int sub_w = (node->width + 1) / 2;
		int sub_h = (node->height + 1) / 2;

		Point lu_start = node->start_xy;
		Point lb_start;
		lb_start.x = node->start_xy.x;
		lb_start.y = node->start_xy.y + sub_h;
		Point ru_start;
		ru_start.x = node->start_xy.x + sub_w;
		ru_start.y = node->start_xy.y;
		Point rb_start;
		rb_start.x = node->start_xy.x + sub_w;
		rb_start.y = node->start_xy.y + sub_h;

		node->child[0] = create_node_(lu_start, sub_w, sub_h, node->depth + 1);
		node->child[1] = create_node_(ru_start, sub_w, sub_h, node->depth + 1);
		node->child[2] = create_node_(lb_start, sub_w, sub_h, node->depth + 1);
		node->child[3] = create_node_(rb_start, sub_w, sub_h, node->depth + 1);
	}

	split_to_depth(node->child[0], max_depth);
	split_to_depth(node->child[1], max_depth);
	split_to_depth(node->child[2], max_depth);
	split_to_depth(node->child[3], max_depth);
}

int Quad_Tree::insert_node_(Quad_Node* node, const cv::Point2f& kp, const int& index)
{
	if (node == nullptr) {
		return -1;
	}
	else {
		if (node->is_leaf) {
			int count = node->points.size();
			if (count + 1 > capacity_) {
				if (node->depth < max_depth_) {
					//std::cout << "----spliting ["<< node->start_xy.x << ", " << node->start_xy.y << ", " << node->width << ", " << node->height << "], depth = " << node->depth << "\n";
					split_node_(node);
					//std::cout << "----split done!----\n";
					return insert_node_(node, kp, index);
				}
				else {
					return -1;
				}
			}
			else {
				node->points.push_back(kp);
				node->indexes.push_back(index);
				//std::cout << "insert (" << kp.pt.x << "," << kp.pt.y << ") in [" << node->start_xy.x << "," << node->start_xy.y << "," << node->width << "," << node->height << "],depth=" << node->depth << "\n";
				return 0;
			}
		}
		else {
			//计算所属区域（2*2网格）坐标-1是处理正好处于边界上的点
			int sub_w = (node->width + 1) / 2, sub_h = (node->height + 1) / 2;
			int area_idx_x = (kp.x - 1 - node->start_xy.x) / sub_w;
			int area_idx_y = (kp.y - 1 - node->start_xy.y) / sub_h;
			int area_idx = 2 * area_idx_y + area_idx_x;
			return insert_node_(node->child[area_idx], kp, index);
		}
	}
}

#if 0
Point kp_coord = { kp.pt.x,kp.pt.y };
//kp_coord.x = kp.pt.x;
//kp_coord.y = kp.pt.y;

Region in_region = find_region(node, kp_coord);
if (in_region == Region::LEFT_UP) {
	return insert_node_(node->lu_child, kp, index);
}
else if (in_region == Region::LEFT_BOTTOM) {
	return insert_node_(node->lb_child, kp, index);
}
else if (in_region == Region::RIGHT_UP) {
	return insert_node_(node->ru_child, kp, index);
}
else if (in_region == Region::RIGHT_BOTTOM) {
	return insert_node_(node->rb_child, kp, index);
}
else {
	return -1;
}
#endif 


bool Quad_Tree::insert_node_r_(Quad_Node * node, const cv::Point2f & kp, const int& index)
{
	if (node == nullptr) {
		return false;
	}
	else {
		if (node->is_leaf) {
			int count = node->points.size();
			if (count + 1 > capacity_) {
				if (node->depth < max_depth_) {
					//std::cout << "----spliting [" << node->start_xy.x << ", " << node->start_xy.y << ", " << node->width << ", " << node->height << "], depth = " << node->depth << "\n";
					split_node_(node);
					//std::cout << "----split done!----\n";
					return insert_node_r_(node, kp, index);
				}
				else {
					return false;
				}
			}
			else {
				for (int i = 0, cnt = node->points.size(); i < cnt; ++i) {
					int xy_dis = abs(kp.x - node->points[i].x) + abs(kp.y - node->points[i].y);
					if (xy_dis < 3) {
						return false;
					}
				}
				node->points.push_back(kp);
				node->indexes.push_back(index);
				//std::cout << "insert (" << kp.pt.x << "," << kp.pt.y << ") in [" << node->start_xy.x << "," << node->start_xy.y << "," << node->width << "," << node->height << "],depth=" << node->depth << "\n";
				return true;
			}
		}
		else {
			//计算所属区域（2*2网格）
			int sub_w = (node->width + 1) / 2, sub_h = (node->height + 1) / 2;
			int area_idx_x = (kp.x - 1 - node->start_xy.x) / sub_w;
			int area_idx_y = (kp.y - 1 - node->start_xy.y) / sub_h;
			int area_idx = 2 * area_idx_y + area_idx_x;
			return insert_node_r_(node->child[area_idx], kp, index);
		}
	}
#if 0
	Point kp_coord;
	kp_coord.x = kp.pt.x;
	kp_coord.y = kp.pt.y;
	Region in_region = find_region(node, kp_coord);
	if (in_region == Region::LEFT_UP) {
		return insert_node_r_(node->lu_child, kp, index);
	}
	else if (in_region == Region::LEFT_BOTTOM) {
		return insert_node_r_(node->lb_child, kp, index);
	}
	else if (in_region == Region::RIGHT_UP) {
		return insert_node_r_(node->ru_child, kp, index);
	}
	else if (in_region == Region::RIGHT_BOTTOM) {
		return insert_node_r_(node->rb_child, kp, index);
	}
	else {
		return false;
	}

	return false;
#endif
}


int Quad_Tree::query_node_(Quad_Node * node, const cv::Point2f & kp)
{
	if (node->is_leaf) {
		int count = node->points.size();
		for (int i = 0; i < count; ++i) {
			int xy_dis = abs(kp.x - node->points[i].x) + abs(kp.y - node->points[i].y);
			//int y_dis = abs(kp.pt.y - node->points[i].pt.y);
			if (xy_dis < 3) {
				return node->indexes[i];
			}
		}
		return -1;
	}

	//计算所属区域（2*2网格）
	int sub_w = (node->width + 1) / 2, sub_h = (node->height + 1) / 2;
	int area_idx_x = (kp.x - 1 - node->start_xy.x) / sub_w;
	int area_idx_y = (kp.y - 1 - node->start_xy.y) / sub_h;
	int area_idx = 2 * area_idx_y + area_idx_x;
	return query_node_(node->child[area_idx], kp);
}

bool cmp_res(const cv::KeyPoint & p1, const cv::KeyPoint & p2)
{
	return p1.response > p2.response;
}

void Quad_Tree::get_near_points(Quad_Node * node, const cv::Point2f & kp, std::vector<int>*kps_indexes)
{
	if (node == nullptr) {
		return;
	}
	else {
		if (node->is_leaf) {
			int count = node->points.size();
			if (count > 0) {
				for (int i = 0; i < count; ++i) {
					kps_indexes->push_back(node->indexes[i]);
				}
			}
			return;
		}
		else {
			//计算所属区域（2*2网格）
			int sub_w = (node->width + 1) / 2, sub_h = (node->height + 1) / 2;
			int area_idx_x = (kp.x - 1 - node->start_xy.x) / sub_w;
			int area_idx_y = (kp.y - 1 - node->start_xy.y) / sub_h;
			int area_idx = 2 * area_idx_y + area_idx_x;
			get_near_points(node->child[area_idx], kp, kps_indexes);
		}
	}
}

#if 0
Region Quad_Tree::find_region(Quad_Node * node, const Point & point)
{
	Region found = Region::OUT_OF_RANGE;
	int x = point.x;
	int y = point.y;
	int sub_w = node->width / 2;
	int sub_h = node->height / 2;
	if (x > node->start_xy.x && x < node->start_xy.x + sub_w) {
		if (y > node->start_xy.y && y < node->start_xy.y + sub_h) {
			found = Region::LEFT_UP;
		}
		else if (y > node->start_xy.y + sub_h && y < node->start_xy.y + node->height) {
			found = Region::LEFT_BOTTOM;
		}
		else {
		}
	}
	else if (x > node->start_xy.x + sub_w && x < node->start_xy.x + node->width) {
		if (y > node->start_xy.y && y < node->start_xy.y + sub_h) {
			found = Region::RIGHT_UP;
		}
		else if (y > node->start_xy.y + sub_h && y < node->start_xy.y + node->height) {
			found = Region::RIGHT_BOTTOM;
		}
		else {
		}
	}
	else {
		found = Region::OUT_OF_RANGE;
	}

	return found;
}
#endif

int Quad_Tree::insert_(const cv::Point2f & kp, const int& index) {
	return insert_node_(root_, kp, index);
}

bool Quad_Tree::insert_r(const cv::Point2f & kp, const int& index)
{
	return insert_node_r_(root_, kp, index);
}

void Quad_Tree::get_hit_points(const cv::Point2f & kp, std::vector<int>*kps_indexes)
{
	get_near_points(root_, kp, kps_indexes);
}

void Quad_Tree::get_node_points(Quad_Node * node, std::vector<cv::Point2f>*filtered)
{
	if (node->is_leaf) {
		int count = node->points.size();
		if (count > 0) {
			for (int i = 0; i < count; ++i) {
				filtered->push_back(node->points[i]);
			}
		}
		return;
	}
	else {
		for (int i = 0; i < 4; ++i) {
			get_node_points(node->child[i], filtered);
		}
	}

}


void Quad_Tree::release_node(Quad_Node * node, const int& depth) {

	if (node == nullptr) {
		return;
	}
	else {
		release_node(node->child[0], depth);
		release_node(node->child[1], depth);
		release_node(node->child[2], depth);
		release_node(node->child[3], depth);
		node->points.clear();
		node->indexes.clear();
		if (node->depth > depth) {
			delete node;
		}
		else if (node->depth == depth) {
			int i = 0;
			node->child[i++] = nullptr;
			node->child[i++] = nullptr;
			node->child[i++] = nullptr;
			node->child[i++] = nullptr;
			node->is_leaf = true;
		}
	}

}

void Quad_Tree::clear_points(Quad_Node * node)
{
	if (node->is_leaf) {
		if (node->points.size() > 0) {
			node->points.clear();
			node->indexes.clear();
		}
		return;
	}
	else {
		clear_points(node->child[0]);
		clear_points(node->child[1]);
		clear_points(node->child[2]);
		clear_points(node->child[3]);
	}
}


void Quad_Tree::get_near_r(Quad_Node * node, const cv::Point2f & kp, std::vector<int>*kps_indexes, const int r)
{
	if (node == nullptr) {
		return;
	}
	else {
		if (node->is_leaf) {
			int count = node->points.size();
			if (count > 0) {
				for (int i = 0; i < count; ++i) {
					float dis = pow((kp.x - node->points[i].x), 2) + pow((kp.y - node->points[i].y), 2);
					if (dis < r * r) {
						kps_indexes->push_back(node->indexes[i]);
					}
				}
			}
			return;
		}
		else {
			//计算所属区域（2*2网格）
			int sub_w = (node->width + 1) / 2, sub_h = (node->height + 1) / 2;
			int area_idx_x = (kp.x - 1 - node->start_xy.x) / sub_w;
			int area_idx_y = (kp.y - 1 - node->start_xy.y) / sub_h;
			int area_idx = 2 * area_idx_y + area_idx_x;
			get_near_r(node->child[area_idx], kp, kps_indexes, r);
		}
	}
}


void Quad_Tree::release(const int& depth) {
	release_node(root_, depth);
}


void Quad_Tree::clear()
{
	clear_points(root_);
}

void Quad_Tree::init_split(const int& depth)
{
	split_to_depth(root_, depth);
}

int Quad_Tree::query_exist_point(const cv::Point2f & kp)
{
	return query_node_(root_, kp);
}

void Quad_Tree::get_kept_kps(std::vector<cv::Point2f>*filtered)
{
	get_node_points(root_, filtered);
}

void Quad_Tree::get_near_r_points(const cv::Point2f & kp, std::vector<int>*kps_indexes, const int r)
{
	get_near_r(root_, kp, kps_indexes, r);
}

int hamming_distance(const cv::Mat & des_l, const cv::Mat & des_r) {

	int dist = 0;
	const int* pa = des_l.ptr<int32_t>();
	const int* pb = des_r.ptr<int32_t>();
//#pragma omp parallel for num_threads(2) reduction(+:dist)
	for (int i = 0; i < 16; i++) {
		unsigned  int v = *(pa+i) ^ *(pb+i);
		v = v - ((v >> 1) & 0x55555555);
		v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
		dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}
	return dist;
}

bool comp(const std::pair<int, int>&a, const std::pair<int, int>&b) {
	return a.first < b.first;
}