# YOLO-IOU

### Dependencies

- OpenCV 4.X

### Usage

- Download the weights file using getWeights.sh
- Change all the files paths.
- Modify CMakeLists.txt (put the path of your opencv instalation path)
- run the following
```bash
mkdir build
cd build
cmake ..
make -j4

./yolo_iou --video=../run.mp4
```

### Project Description

- This is using opencv's dnn module for YOLO. IOU tracker and YOLO are combined together in cpp.

