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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "lvt_system.h"
#include <opencv2/opencv.hpp>
#include<windows.h>

static void dump_kitti_trajectory(const std::string &fileName, const lvt_pose_array &seq_poses)
{
    std::cout << "saving trajectory " << fileName << "\n";
    std::ofstream file(fileName.c_str());
    assert(file.is_open());
    file << std::fixed;
    for (size_t i = 0, count = seq_poses.size(); i < count; i++)
    {
        lvt_matrix33 r = seq_poses[i].get_orientation_matrix();
        lvt_vector3 pos = seq_poses[i].get_position();
        file << std::setprecision(9) << r(0, 0) << " " << r(0, 1) << " " << r(0, 2) << " " << pos.x() << " "
             << r(1, 0) << " " << r(1, 1) << " " << r(1, 2) << " " << pos.y() << " "
             << r(2, 0) << " " << r(2, 1) << " " << r(2, 2) << " " << pos.z() << std::endl;
    }
    file.close();
}

int run_test(int seq_num) {
    //定义路径
    std::string lvt_test_prefix = "D:\\Research\\openSrc\\lvt\\examples\\kitti";
    std::string seq_folder_prefix = "D:\\Research\\openSrc\\lvt\\datasets\\KITTI\\dataset\\sequences\\";
    char seq_cstr[5];
    std::sprintf(seq_cstr, "%02d", seq_num);
    std::string seq_str = std::string(seq_cstr);
    std::string seq_foldler_path = seq_folder_prefix + seq_str;
    std::string img_path_left = "\\image_0\\%06d.png";
    std::string img_path_right = "\\image_1\\%06d.png";
    std::string capL_str = seq_foldler_path + img_path_left;    //左相机读取路径
    std::string calR_str = seq_foldler_path + img_path_right;   //右相机读取路径

    //初始化相机捕获对象
    cv::VideoCapture capL(capL_str);
    cv::VideoCapture capR(calR_str);
    if (!capL.isOpened() || !capR.isOpened())
    {
        std::cout << "failed to get image sequences" << std::endl;
        return -1;
    }

    //读取相机标定文件
    std::string cam_fname = lvt_test_prefix + std::string("\\calib\\") + seq_str + ".yml";
    cv::FileStorage cam_fs(cam_fname, cv::FileStorage::READ);
    if (!cam_fs.isOpened())
    {
        std::cout << "failed to open camera matrix yml file" << std::endl;
        return -1;
    }

    //读取LVT配置的参数，包括算法各种阈值和策略
    lvt_parameters params;
    std::string vo_cfg_str = lvt_test_prefix + std::string("\\vo_config.yaml");
    const char* vo_cfg_file = vo_cfg_str.c_str();
    if (!params.init_from_file(vo_cfg_file))
    {
        std::cout << "failed to initialize from vo_config.yml file." << std::endl;
        return -1;
    }

    //从标定文件读取相机内参
    cv::Mat cam_mtrx;
    cam_fs["camera_matrix"] >> cam_mtrx;
    double baseline = cam_fs["baseline"];
    cam_fs.release();

    int frameCount = (int)capL.get(cv::CAP_PROP_FRAME_COUNT);    //总帧数
    int w = (int)capL.get(cv::CAP_PROP_FRAME_WIDTH);             //图像宽
    int h = (int)capL.get(cv::CAP_PROP_FRAME_HEIGHT);            //图像高

    //读取相机内参
    params.fx = cam_mtrx.at<double>(0, 0);
    params.fy = cam_mtrx.at<double>(1, 1);
    params.cx = cam_mtrx.at<double>(0, 2);
    params.cy = cam_mtrx.at<double>(1, 2);
    params.baseline = baseline;
    params.img_width = w;
    params.img_height = h;

    //创建VO系统对象
    lvt_system* vo = lvt_system::create(params, lvt_system::eSensor_STEREO);

    lvt_pose_array seq_poses;    //位姿数组
    seq_poses.resize(frameCount);
    std::vector<double> frame_exec_times(frameCount, 0.0);
    std::string kitti_out_file_name = seq_str + std::string(".txt");    //结果文件
    const double tick_f = cv::getTickFrequency();    //准备计时
    for (int i = 0; i < frameCount; i++)
    {
        std::cout << "frame #" << i << " of sequence " << seq_cstr << "\n";
        cv::Mat imgLeft, imgRight;
        capL >> imgLeft;
        capR >> imgRight;
        if (imgLeft.channels() != 1)
        {
            cv::cvtColor(imgLeft, imgLeft, cv::COLOR_BGR2GRAY);
        }
        if (imgRight.channels() != 1)
        {
            cv::cvtColor(imgRight, imgRight, cv::COLOR_BGR2GRAY);
        }

        //预处理
        //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        //clahe->setClipLimit(4);
        //cv::Mat clahe_left, clahe_right;
        //clahe->apply(imgLeft, clahe_left);
        //clahe->apply(imgRight, clahe_right);

        double t = (double)cv::getTickCount();
        seq_poses[i] = vo->track(imgLeft, imgRight);

        //std::cout << "current position:";
        //std::cout << seq_poses[i].get_position() << "\n";
        frame_exec_times[i] = (((double)cv::getTickCount() - t) / tick_f);

        if (vo->get_state() == lvt_system::eState_LOST ||
            vo->should_quit())
        {
            std::cout << "Sequence " << seq_str << " test failed tracking.\n";
            break;
        }

    }

    dump_kitti_trajectory(kitti_out_file_name, seq_poses);
    lvt_system::destroy(vo);

    double total_time = 0.0;
    for (size_t i = 0; i < frameCount; i++)
    {
        total_time += frame_exec_times[i];
    }

    std::cout << "Average frame processing time: " << total_time / double(frameCount) << std::endl;

    return 0;
}

int main(int argc, char **argv)
{
    std::vector<int> seq;
#if 1
    //seq.push_back(2);
    //seq.push_back(10);
    //seq.push_back(6);
    //seq.push_back(9);
    seq.push_back(4);
    //seq.push_back(3);
    //seq.push_back(5);
    //seq.push_back(7);
    //seq.push_back(8);
    //seq.push_back(0);
    for (int i = 0; i < seq.size(); ++i) {
        run_test(seq[i]);
        std::cout << "test of sequence " << seq[i] << " finished, starting next test...\n";
        Sleep(2000);
    }
#endif

#if 0
    if (argc != 3)
    {
        std::cout << "Usage ./kitti_example sequences_dir seq_number" << std::endl;
        return -1;
    }

    std::string exec_prefix = "D:\\Research\\openSrc\\lvt\\examples\\kitti";    //绝对路径

    int seq_num = atoi(argv[2]);
    char seq_cstr[5];
    std::sprintf(seq_cstr, "%02d", seq_num);
    std::string seq_str = std::string(seq_cstr);
    std::string dir_prefix = std::string(argv[1]) + std::string("\\") + seq_str;
    //std::string dir_postfix_left = "/image_0/%06d.png";
    //std::string dir_postfix_right = "/image_1/%06d.png";
    std::string dir_postfix_left = "\\image_0\\%06d.png";    //for windows
    std::string dir_postfix_right = "\\image_1\\%06d.png";
    std::string capL_str = dir_prefix + dir_postfix_left;
    std::string calR_str = dir_prefix + dir_postfix_right;

    cv::VideoCapture capL(capL_str);
    cv::VideoCapture capR(calR_str);
    if (!capL.isOpened() || !capR.isOpened())
    {
        std::cout << "failed to get image sequences" << std::endl;
        return -1;
    }

    //相机标定文件
    std::string cam_fname = exec_prefix + std::string("\\calib\\") + seq_str + ".yml";
    cv::FileStorage cam_fs(cam_fname, cv::FileStorage::READ);
    if (!cam_fs.isOpened())
    {
        std::cout << "failed to open camera matrix yml file" << std::endl;
        return -1;
    }

    //VO系统参数全部写在vo_config.yaml，包括内部算法的各种阈值，策略
    lvt_parameters params;
    std::string vo_cfg_str = exec_prefix + std::string("\\vo_config.yaml");
    const char* vo_cfg_file = vo_cfg_str.c_str();
    if (!params.init_from_file(vo_cfg_file))
    {
        std::cout << "failed to initialize from vo_config.yml file." << std::endl;
        return -1;
    }

    //从标定文件读取相机内参
    cv::Mat cam_mtrx;
    cam_fs["camera_matrix"] >> cam_mtrx;
    double baseline = cam_fs["baseline"];
    cam_fs.release();

    int frameCount = (int)capL.get(cv::CAP_PROP_FRAME_COUNT);    //总帧数
    int w = (int)capL.get(cv::CAP_PROP_FRAME_WIDTH);             //图像宽
    int h = (int)capL.get(cv::CAP_PROP_FRAME_HEIGHT);            //图像高

    //VO系统参数包括相机内参，也读进去
    params.fx = cam_mtrx.at<double>(0, 0);
    params.fy = cam_mtrx.at<double>(1, 1);
    params.cx = cam_mtrx.at<double>(0, 2);
    params.cy = cam_mtrx.at<double>(1, 2);
    params.baseline = baseline;
    params.img_width = w;
    params.img_height = h;

    //创建VO系统对象
    lvt_system *vo = lvt_system::create(params, lvt_system::eSensor_STEREO);

    lvt_pose_array seq_poses;    //位姿数组
    seq_poses.resize(frameCount);
    std::vector<double> frame_exec_times(frameCount, 0.0);
    std::string kitti_out_file_name = seq_str + std::string(".txt");    //结果文件
    const double tick_f = cv::getTickFrequency();    //准备计时
    for (int i = 0; i < frameCount; i++)
    {
        //std::cout << "frame number " << i << "...\n";
        //std::cout << "Frame number: " << i << "/" << frameCount << "\r";//<< std::flush;

        cv::Mat imgLeft, imgRight;
        capL >> imgLeft;
        capR >> imgRight;
        if (imgLeft.channels() != 1)
        {
            cv::cvtColor(imgLeft, imgLeft, cv::COLOR_BGR2GRAY);
        }
        if (imgRight.channels() != 1)
        {
            cv::cvtColor(imgRight, imgRight, cv::COLOR_BGR2GRAY);
        }
        
        //预处理
        //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        //clahe->setClipLimit(4);
        //cv::Mat clahe_left, clahe_right;
        //clahe->apply(imgLeft, clahe_left);
        //clahe->apply(imgRight, clahe_right);

        double t = (double)cv::getTickCount();
        seq_poses[i] = vo->track(imgLeft, imgRight);

        //std::cout << "current position:";
        //std::cout << seq_poses[i].get_position() << "\n";
        frame_exec_times[i] = (((double)cv::getTickCount() - t) / tick_f);

        if (vo->get_state() == lvt_system::eState_LOST ||
            vo->should_quit())
        {
            break;
        }

    }

    dump_kitti_trajectory(kitti_out_file_name, seq_poses);
    lvt_system::destroy(vo);

    double total_time = 0.0;
    for (size_t i = 0; i < frameCount; i++)
    {
        total_time += frame_exec_times[i];
    }

    std::cout << "Average frame processing time: " << total_time / double(frameCount) << std::endl;

    return 0;
#endif
}
