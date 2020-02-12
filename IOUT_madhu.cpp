
#include <iostream>
#include <vector>
#include <algorithm>

// struct for bounding box
struct Boundingbox
{
	float x;
	float y;
	float w;
	float h;
	float score;
};

// struct for track
struct Track
{
	std::vector<Boundingbox> boxes;
	float max_score;
	int start_frame;
	int track_id;
};

// function to find iou
float find_IOU(Boundingbox box1, Boundingbox box2)
{
	// convert xywh to x1y1x2y2
	float min_x_1 = box1.x;
	float min_y_1 = box1.y;
	float max_x_1 = box1.x + box1.w;
	float max_y_1 = box1.y + box1.h;

	float min_x_2 = box2.x;
	float min_y_2 = box2.y;
	float max_x_2 = box2.x + box2.w;
	float max_y_2 = box2.y + box2.h;

	// if boxes don't overlap, return 0
	if (min_x_1 > max_x_2 || max_x_1 < min_x_1 || min_y_1 > max_y_2 || max_y_1 < min_y_2)
		return 0;
	else
	{
		// find iou
		float length = std::min(max_x_2, max_x_1) - std::min(min_x_2, min_x_1);
		float width = std::min(max_y_2, max_y_1) - std::min(min_y_2, min_y_1);
		
		float intersection = length * width;

		float box1_area = (max_x_1 - min_x_1) * (max_y_1 - min_y_1);
		float box2_area = (max_x_2 - min_x_2) * (max_y_2 - min_y_2);

		float union_ = box1_area + box2_area - intersection;

		float iou = intersection/union_;

		return iou;
	}

}

// highest iou 
int Highest_iou (Boundingbox box1, std::vector<Boundingbox> boxes)
{
	float highest = 0;
	int index = -1;
	for (int i = 0; i < boxes.size(); i++)
	{
		float iou_box = find_IOU(box1, boxes[i]);
		
		if (iou_box > highest)
		{
			highest = iou_box;
			index = i;
		}
	}
	return index;
}

std::vector< Track > IOU_tracker(float sigma_l, float sigma_h, float sigma_iou, int t_min, std::vector< std::vector<Boundingbox> > detections)
{
	
	int index;
	bool updated;

	// initiate active tracks, finished tracks
	std::vector<Track> active_tracks;
	std::vector<Track> finished_tracks;

	// do for all the frames
	for (int frame_no = 0; frame_no < detections.size(); frame_no++)
	{
		// get detections from the frame
		std::vector<Boundingbox> frame_boxes = detections[frame_no];

		// for each track in active tracks
		for (int i = 0; i < active_tracks.size(); i++)
		{
			Track track = active_tracks[i];
			updated = false;
			// the index of box with highest iou
			index = Highest_iou(track.boxes.back(), frame_boxes);

			// if no box is found or 
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

		// add new tracks in active tracks
		for ( auto box : frame_boxes)
		{
			std::vector<Boundingbox> new_box;

			new_box.push_back(box);

			Track t = {new_box, box.score, frame_no, 0};
			
			active_tracks.push_back(t);
		}

		// creating new tracks
		for (auto track : active_tracks)
		{
			if (track.max_score >= sigma_h && track.boxes.size() >= t_min)
			{
				finished_tracks.push_back(track);
			}
		}

		return finished_tracks;

	}
}

