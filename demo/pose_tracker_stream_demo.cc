﻿#include <stdlib.h>
#ifndef _WIN32
#include <sys/prctl.h>  // for: prctl
#endif
#include <unistd.h>  // for: getopt

#include <algorithm>  // for: swap
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "dataloader.hpp"
#include "opencv2/opencv.hpp"
#include "pose_estimation.hpp"
#include "task/vision/object_detection_task.h"
#include "task/vision/pose_estimation_task.h"
#ifdef DEBUG
#include "utils/time.h"
#endif

#include "utils/utils.h"

void setThreadName(const char* name) {
#ifndef _WIN32
  prctl(PR_SET_NAME, name);
#endif
}

class Tracker {
 public:
  Tracker(const std::string& det_file_path, const std::string& pose_file_path) {
    det_file_path_ = det_file_path;
    pose_file_path_ = pose_file_path;
  }
  Tracker(const ObjectDetectionOption& detection_option,
          const PoseEstimationOption& estimation_option) {
    detection_option_ = detection_option;
    estimation_option_ = estimation_option;
  }
  ~Tracker() {}
  // 初始化/反初始化
  int init() {
    if (!det_file_path_.empty()) {
      objectdetectiontask_ = std::unique_ptr<ObjectDetectionTask>(
          new ObjectDetectionTask(det_file_path_));
      poseestimationtask_ = std::unique_ptr<PoseEstimationTask>(
          new PoseEstimationTask(pose_file_path_));
    } else {
      objectdetectiontask_ = std::unique_ptr<ObjectDetectionTask>(
          new ObjectDetectionTask(detection_option_));
      poseestimationtask_ = std::unique_ptr<PoseEstimationTask>(
          new PoseEstimationTask(estimation_option_));
    }
    return getInitFlag();
  }

  int getInitFlag() {
    return (objectdetectiontask_->getInitFlag() ||
            poseestimationtask_->getInitFlag());
  }

  int uninit() { return 0; }

  // 推理
  int infer(cv::Mat frame) {
    if (frame.empty()) {
      return -1;
    }
    ObjectDetectionResult objs_temp = objectdetectiontask_->Detect(frame);
    int count{0}, flag{0};
    for (count = 0; count < static_cast<int>(objs_temp.result_bboxes.size());
         count++) {
      if (objs_temp.result_bboxes[count].label == 0) {
        flag = 1;
        break;
      }
    }
    if (flag) {
      PoseEstimationResult poses_temp =
          poseestimationtask_->Estimate(frame, objs_temp.result_bboxes[count]);
      poses_mutex_.lock();
      poses_array_.push(poses_temp);  // 直接替换掉当前的 objs_array_
      poses_mutex_.unlock();
      return poses_array_.size();
    } else {
      return 0;
    }
  }

  // 查询检测结果
  int estimated() { return poses_array_.size(); }

  // 移走检测结果
  struct PoseEstimationResult getPose() {
    struct PoseEstimationResult poses_moved;
    poses_mutex_.lock();
    poses_moved = poses_array_.back();
    std::queue<struct PoseEstimationResult> empty;
    std::swap(empty, poses_array_);
    poses_mutex_.unlock();
    return poses_moved;
  }

 private:
  std::mutex poses_mutex_;
  std::queue<struct PoseEstimationResult> poses_array_;
  std::unique_ptr<ObjectDetectionTask> objectdetectiontask_;
  std::unique_ptr<PoseEstimationTask> poseestimationtask_;
  std::string pose_file_path_;
  std::string det_file_path_;
  ObjectDetectionOption detection_option_;
  PoseEstimationOption estimation_option_;
};

void Inference(DataLoader& dataloader, Tracker& tracker) {
  setThreadName("TrackerThread");
  cv::Mat frame;
  while (dataloader.ifEnable()) {
    auto start = std::chrono::steady_clock::now();
    if (!dataloader.isUpdated()) {
      continue;
    }
    frame = dataloader.peekFrame();  // 取(拷贝)一帧数据
    if ((frame).empty()) {
      dataloader.setDisable();
      break;
    }
    int flag = tracker.infer(frame);  // 推理并保存检测结果
    auto end = std::chrono::steady_clock::now();
    auto detection_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    dataloader.setDetectionFps(1000 / (detection_duration.count()));
    if (flag == 0) {
      std::cout << "[ WARNING ] Unable to catch person" << std::endl;  // 无人
    }
    if (flag == -1) {
      std::cout << "[ ERROR ] Infer frame failed" << std::endl;
      break;  // 摄像头结束拍摄或者故障
    }
  }
}

// 检测线程
void Track(DataLoader& dataloader, Tracker& tracker) {
  setThreadName("OnnxruntimeThread");
  if (tracker.init() != 0) {
    std::cout << "[ ERROR ] Tracker init error" << std::endl;
    return;
  }
  std::thread t1(Inference, std::ref(dataloader), std::ref(tracker));
  t1.join();
  std::cout << "Track thread quit" << std::endl;
}

// 预览线程
void Preview(DataLoader& dataloader, Tracker& tracker) {
  cv::Mat frame;
  PoseEstimationResult poses;
  auto now = std::chrono::steady_clock::now();
  poses.timestamp = now;
  int count = 0;
  int dur = 0;
  int enable_show = 1;
  const char* showfps = getenv("SUPPORT_SHOWFPS");
  const char* show = getenv("SUPPORT_SHOW");
  if (show && strcmp(show, "-1") == 0) {
    enable_show = -1;
  }
  while (dataloader.ifEnable()) {
    auto start = std::chrono::steady_clock::now();
    frame = dataloader.fetchFrame();  // 取(搬走)一帧数据
    if ((frame).empty()) {
      dataloader.setDisable();
      break;
    }
    if (tracker.estimated())  // 判断原因: detector.detected 不用锁,
                              // detector.get_object 需要锁;
    {
      // 是否有检测结果
      poses = tracker.getPose();  // 取(搬走)检测结果(移动赋值)
      if (poses.result_points.size()) {
        int input_height = dataloader.getResizeHeight();
        int input_width = dataloader.getResizeWidth();
        int img_height = frame.rows;
        int img_width = frame.cols;
        float resize_ratio = std::min(
            static_cast<float>(input_height) / static_cast<float>(img_height),
            static_cast<float>(input_width) / static_cast<float>(img_width));
        float dw = (input_width - resize_ratio * img_width) / 2;
        float dh = (input_height - resize_ratio * img_height) / 2;
        for (size_t i = 0; i < poses.result_points.size(); i++) {
          poses.result_points[i].x =
              (poses.result_points[i].x - dw) / resize_ratio;
          poses.result_points[i].y =
              (poses.result_points[i].y - dh) / resize_ratio;
        }
      }
    }
    // 调用 detector.detected 和 detector.get_object 期间,
    // 检测结果依然可能被刷新
    now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - poses.timestamp);
    if (duration.count() < 1000 && poses.result_points.size()) {
      draw_points_inplace((frame), poses.result_points);  // 画框
    }
    int preview_fps = dataloader.getPreviewFps();
    int detection_fps = dataloader.getDetectionFps();
    if (showfps != nullptr) {
      cv::putText(frame, "preview fps: " + std::to_string(preview_fps),
                  cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5f,
                  cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
      cv::putText(frame, "detection fps: " + std::to_string(detection_fps),
                  cv::Point(500, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5f,
                  cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    if (enable_show != -1) {
      cv::imshow("Track", (frame));
      cv::waitKey(10);
    }
    auto end = std::chrono::steady_clock::now();
    auto preview_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    count++;
    dur = dur + preview_duration.count();
    if (dur >= 1000) {
      dataloader.setPreviewFps(count);
      dur = 0;
      count = 0;
    }
    if (enable_show != -1) {
      if (cv::getWindowProperty("Track", cv::WND_PROP_VISIBLE) < 1) {
        dataloader.setDisable();
        break;
      }
    }
  }
  std::cout << "Preview thread quit" << std::endl;
  if (enable_show != -1) {
    cv::destroyAllWindows();
  }
}

int main(int argc, char* argv[]) {
  std::string det_file_path, pose_file_path, input, input_type;
  int resize_height{320}, resize_width{320};
  ObjectDetectionOption detection_option;
  PoseEstimationOption estimation_option;
  std::unique_ptr<Tracker> tracker;
  int o;
  const char* optstring = "w:h:";
  while ((o = getopt(argc, argv, optstring)) != -1) {
    switch (o) {
      case 'w':
        resize_width = atoi(optarg);
        break;
      case 'h':
        resize_height = atoi(optarg);
        break;
      case '?':
        std::cout << "[ ERROR ] Unsupported usage" << std::endl;
        break;
    }
  }
  if (argc - optind == 4) {
    det_file_path = argv[optind];
    pose_file_path = argv[optind + 1];
    input = argv[optind + 2];
    input_type = argv[optind + 3];
    tracker =
        std::unique_ptr<Tracker>(new Tracker(det_file_path, pose_file_path));
  } else if (argc - optind == 5) {
    detection_option.model_path = argv[optind];
    detection_option.label_path = argv[optind + 1];
    estimation_option.model_path = argv[optind + 2];
    input = argv[optind + 3];
    input_type = argv[optind + 4];
    tracker = std::unique_ptr<Tracker>(
        new Tracker(detection_option, estimation_option));
  } else {
    std::cout
        << "Please run with " << argv[0]
        << " <det_model_file_path> <det_label_file_path> "
           "<pose_model_file_path> <input> <input_type> (video or cameraId "
           "option(-h <resize_height>) option(-w <resize_width>) or "
        << argv[0]
        << " <det_config_file_path> <pose_config_file_path> <input> "
           "<input_type> (video or cameraId option(-h <resize_height>) "
           "option(-w <resize_width>)"
        << std::endl;
    return -1;
  }
  SharedDataLoader dataloader{resize_height, resize_width};
  if (dataloader.init(input) != 0) {
    std::cout << "[ ERROR ] dataloader init error" << std::endl;
    return -1;
  }

  std::thread t(Track, std::ref(dataloader), std::ref(*tracker));
  setThreadName("PreviewThread");
  Preview(dataloader, *tracker);
  t.join();
  return 0;
}
