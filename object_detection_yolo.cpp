// Usage example: go to build -> ./yolo_iou --video=../run.mp4

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "IOUT_madhu.cpp"

//using namespace cv;
//using namespace dnn;
//using namespace std;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of input image
int inpHeight = 416; // Height of input image

// classes, a vector containing strings
std::vector<std::string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
std::vector<Boundingbox> postprocess(cv::Mat& frame, const std::vector<cv::Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

// Get the names of the output layers
std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);

// Draw the tracked bounding box
void drawtrack(Boundingbox tbox, cv::Mat& frame);

// get the points for visualization
cv::Point trackable_points(Boundingbox box);


// initiate active tracks and finished tracks
std::vector<Track> active_tracks;
std::vector<Track> finished_tracks;
  
// tracker thresholds   
//float sigma_l = 0.0;
float sigma_h = 0.5;
float sigma_iou = 0.5;
int t_min = 3;
int frame_no = 0;


int main()
{
    // Load names of classes
    std::string classesFile = "/home/madhu/c++/YOLO-IOU/network_files/coco.names";
    // Read from the file
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (std::getline(ifs, line)) classes.push_back(line);
    
    // Give the configuration and weight files for the model
    std::string modelConfiguration = "/home/madhu/c++/YOLO-IOU/network_files/yolov3.cfg";
    std::string modelWeights = "/home/madhu/c++/YOLO-IOU/network_files/yolov3.weights";

    // Load the network  
    cv::dnn::Net net = cv::dnn::readNet(modelConfiguration, modelWeights);  //readNetFromDarknet for opencv 3.4.x

    // define computational parameters
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    // Open a video file or an image file or a camera stream.
    std::string str, outputFile;

    std::string video_file = "/home/madhu/c++/YOLO-IOU/videos/car_trim.mp4";
    

    cv::VideoCapture cap;
    cap.open(video_file);

    outputFile = video_file.replace(video_file.end()-4, video_file.end(), "_yolo_out_cpp.avi");    
    
    //std::cout<<outputFile<<std::endl;
    
    cv::VideoWriter video; 
    cv::Mat frame, blob;
    
    bool updated;
    int index;

    // Get the video writer initialized to save the output video
    video.open(outputFile, cv::VideoWriter::fourcc('M','J','P','G'), 28, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        
    // Create a window
    static const std::string kWinName = "Object detection and IOU tracking with OpenCV";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

    // Process frames.
    while (cv::waitKey(1) < 0)
    //for(;;)
    {
        // get frame from the video
        cap >> frame;

        frame_no+=1;

        // Stop the program if reached end of video
        if (frame.empty()) {
            std::cout << "Done processing !!!" << std::endl;
            std::cout << "Output file is stored as " << outputFile << std::endl;
            cv::waitKey(3000);
            break;
        }
        // Create a 4D blob from a frame. // cvSize was there instead of Size
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
        
        //Sets the input to the network
        net.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        std::vector<cv::Mat> outs;

        net.forward(outs, getOutputsNames(net));
        
        // Remove the bounding boxes with low confidence
        std::vector<Boundingbox> frame_boxes = postprocess(frame, outs);

        //print the boxes for visulaization

        std::cout<<"detected boxes are : "<< std::endl;
        
        for (Boundingbox box: frame_boxes)
        {
            std::cout<<"X : "<< box.x<<", Y : "<< box.y <<", width : "<< box.w<<", Height : "<<box.h<<", Score : "<<box.score<<std::endl;

        }

        // tracking code goes here

        std::vector<cv::Point> points;
        
        // add new tracks in active tracks
        for ( auto box : frame_boxes)
        {
            std::vector<Boundingbox> new_box;

            new_box.push_back(box);

            Track t = {new_box, box.score, frame_no, 0};
            
            active_tracks.push_back(t);
        }


        // for each track in active tracks
        for (int i = 0; i < active_tracks.size(); i++)
        {
            Track track = active_tracks[i];


            updated = false;
            // the index of box with highest iou
            index = Highest_iou(track.boxes.back(), frame_boxes);

            // if box is found and its iou greater than sigma_iou 
            if (index != -1 && find_IOU(track.boxes.back(), frame_boxes[index]) >= sigma_iou)
            {
                track.boxes.push_back(frame_boxes[index]);

                if (track.max_score < frame_boxes[index].score)
                {
                    // update the score in tracks
                    track.max_score = frame_boxes[index].score;
                }
                frame_boxes.erase(frame_boxes.begin() + index);

                // updating the track
                active_tracks[i] = track;
                updated = true;

                drawtrack(track.boxes.back(), frame);

                //points.push_back(cv::Point(active_tracks[i].boxes[0].x + (active_tracks[i].boxes[0].w/2), 
                //	active_tracks[i].boxes[0].y + (active_tracks[i].boxes[0].h))); 
                
                std::cout<<"success!!"<<std::endl;
                std::cout<<"size "<<active_tracks[i].boxes.size()<<std::endl;
                //cv::Point point_x =  active_tracks[i].boxes[0].x + (active_tracks[i].boxes[0].w/2);
                //cv::Point point_y =  active_tracks[i].boxes[0].y + (active_tracks[i].boxes[0].h);

                //cv::line(frame, cv::Point(active_tracks[i].boxes[0].x, active_tracks[i].boxes[0].y), 
                //	cv::Point(active_tracks[i].boxes[0].x, active_tracks[i].boxes[0].y), cv::Scalar(0, 0, 255), 2);
                //cv::circle(frame, cv::Point(50,50), 3, cv::Scalar(0,0,255), 2);
                

                //cv::circle(frame, trackable_points(track.boxes.back()), 3, cv::Scalar(0,0,255), 2);
                int a;
                if (track.boxes.size()>20)
                	a = track.boxes.size() - 20;
                else a = 0;

                for (i = a; i < track.boxes.size(); i++)
                {
               		cv::circle(frame,trackable_points(track.boxes[i]), 3, cv::Scalar(10,0,255),2);
               	}
             
                //if (track.boxes.size() > 3)
                //{
                //	int second_last = track.boxes.size()-2;
                //	int third_last = track.boxes.size()-3;
                	//cv::line(frame, cv::Point((track.boxes.back().x + (track.boxes.back().w/2), track.boxes.back().y + (track.boxes.back().h))), 
                     //	cv::Point(track.boxes[second_last].x + (track.boxes[second_last].w/2), track.boxes[second_last].y + (track.boxes[second_last].h)), cv::Scalar(0, 0, 255), 2);
                	

                	//cv::circle(frame, trackable_points(track.boxes[second_last]), 3, cv::Scalar(0,255,0), 2);

                	//cv::circle(frame, trackable_points(track.boxes[third_last]), 3, cv::Scalar(0,255,0), 1);
                		
                		
                	//cv::line(frame, trackable_points(track.boxes.back()), 
                     //	trackable_points(track.boxes[second_last]), cv::Scalar(0, 0, 255), 2);
                //}
            }

            // if not updated, append them into finished tracks
            if (!updated)
            {
                if (track.max_score > sigma_h && track.boxes.size() > t_min)
                {
                    finished_tracks.push_back(track);   
                }
                // 
                active_tracks.erase(active_tracks.begin() + i);

                i--;
            }
        }



      	//for (int j = 1; j < frame_boxes.size(); j++)
     	//{
     	//	for (size_t i = 1; i < active_tracks.size(); i++)
     	//	{
     	//		cv::line(frame, cv::Point(active_tracks[i-1].boxes[j].x, active_tracks[i-1].boxes[j].y), cv::Point(active_tracks[i].boxes[j].x, active_tracks[i].boxes[j].y), cv::Scalar(0, 0, 255), 2);
     	//	}
        //}

        // +
        for (auto track : active_tracks)
        {

            if (track.max_score >= sigma_h && track.boxes.size() >= t_min)
            {
                finished_tracks.push_back(track);
            }
        }

        // yolo code 
        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000; //  Hzs (it measures in Milli Hzs)
        double t = net.getPerfProfile(layersTimes) / freq; //seconds

        std::string label = cv::format("Inference time for a frame : %.2f ms", t);
        cv::putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        

        std::ostringstream str1;             // initiate a string
        str1 << active_tracks.size();        // send the number to the string
        std::string number = str1.str();       // convert the number to string 

        std::string label_active_tracks = "No of active_tracks : "+ number;
        std::cout<< label_active_tracks<< std::endl;
        cv::putText(frame, label_active_tracks, cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        
        // Write the frame with the detection boxes
        cv::Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U); // undo pre processing

        video.write(detectedFrame);
        
        cv::imshow(kWinName, frame);
        //std::cout<< "done!!"<<frame_no<< std::endl;
        
    }
    
    cap.release();
    video.release();

    //std::cout<<" total number of tracks happened : "<<finished_tracks.size()<<std::endl;

    return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
std::vector<Boundingbox> postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // initiate
    std::vector<Boundingbox> boundingboxes;
    
    //std::cout<<outs.size()<<std::endl; // outs.size() = 3 (represents 3 scales)
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        
        float* data = (float*)outs[i].data;
        // rows and columns to represent grid cells
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols); // colRange(start, end)
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));

            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
               box.x + box.width, box.y + box.height, frame);
    
        float x = box.x;
        float y = box.y;
        float w = box.width;
        float h = box.height;
        float score = confidences[idx];

        boundingboxes.push_back(Boundingbox{x, y, w, h, score}); // use {} for inserting elements into empty struct
    }

    return boundingboxes;
}


// Draw the tracked bounding box
void drawtrack(Boundingbox tbox, cv::Mat& frame)
{
    int left = tbox.x;
    int top = tbox.y;
    int right = tbox.x + tbox.w;
    int bottom = tbox.y + tbox.h;
    float score = tbox.score;


    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(100, 100, 50), 2);
    
    //Get the label and its confidence
    std::string label = cv::format("%.2f", score);
    label = "tracked with score " + label ;

    
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1*labelSize.height)), cv::Point(left + round(1*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0),1);
}










// Get the names of the output layers
std::vector<std::string> getOutputsNames(const cv::dnn::Net& net)
{
    static std::vector<std::string> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}



cv::Point trackable_points(Boundingbox box)
{

	// 
	float point_x = box.x + (box.w/2);
	float point_y = box.y + box.h; 
	
	cv::Point points_track = cv::Point(point_x, point_y);
	return points_track;
}


// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1*labelSize.height)), cv::Point(left + round(1*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0),1);
}

