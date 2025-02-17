import os
import cv2
from ultralytics import YOLO

# model preparing
model_path = r'best.pt'
model = YOLO(model_path)
print(model.names)
# ----------------------------------
# get the video
video_path = r'D:\pycharm\cofee\videos and images\6444194-uhd_3840_2160_24fps.mp4'
cap = cv2.VideoCapture(video_path)
# ----------------------------------
# line coordinates
line_coordinates = [
    (2265, 717, 2896, 962)]

# ----------------------------------
# colors
red = (0, 0, 255)
black = (59, 62, 64)
white = (255,255,255)
offwhite = (219, 226, 231)
detection_color = (214, 189, 106)
# ----------------------------------
# initialize the helpful dic and variables in counting process
coffee_package_counting = 0
track_ids = {}
track_positions = {}  # To store positions for the track tail
# ----------------------------------

orginal_video_name = os.path.basename(video_path)
out_video_name = f'detected__{orginal_video_name}'
out_path = os.path.join(r'D:\pycharm\cofee', out_video_name)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
# ----------------------------------

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_count = 0

if not cap.isOpened():
    print('video not found')
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update progress
    frame_count += 1
    progress = (frame_count / total_frames) * 100
    print(f"\rProcessing: {progress:.1f}% ({frame_count}/{total_frames})", end="")

    results = model.track(frame, persist=True)[0]

    for coord in line_coordinates:
        x1_coor, y1_coor, x2_coor, y2_coor = coord
        cv2.line(frame, (x1_coor, y1_coor), (x2_coor, y2_coor), red, 10)
        cv2.circle(frame, (x1_coor, y1_coor), 15, red, -1)
        cv2.circle(frame, (x2_coor, y2_coor), 15, red, -1)

    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        track_id = int(result.id[0]) if result.id is not None else None

        class_idx = int(result.cls[0]) if result.cls is not None else None
        class_name = model.names[class_idx] if class_idx is not None else "Unknown"

        center_x = (x2 + x1) // 2
        center_y = (y2 + y1) // 2

        lane1_x_min, lane1_x_max = 2265, 2896

        line_x1, line_y1, line_x2, line_y2 = line_coordinates[0]
        if lane1_x_min <= center_x <= lane1_x_max and (line_y1 - 5 <= center_y <= line_y2 + 5):
            if track_id not in track_ids:
                track_ids[track_id] = True
                coffee_package_counting += 1

        # Track tail drawing
        if track_id not in track_positions:
            track_positions[track_id] = []  # Initialize a new list for storing positions

        # Append the new position to the list
        track_positions[track_id].append((center_x, center_y))

        # Draw the track tail (lines connecting previous positions)
        for i in range(1, len(track_positions[track_id])):
            cv2.line(frame, track_positions[track_id][i - 1], track_positions[track_id][i], detection_color, 8)

        # Limit the length of the tail to prevent memory overflow
        if len(track_positions[track_id]) > 20:
            track_positions[track_id].pop(0)

        # Draw the bounding box and ID label
        cv2.rectangle(frame, (x1, y1), (x2, y2), detection_color, 7)

        label = f'ID {track_id} {class_name}'
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
        rect_start = (x1, y1 - text_height - baseline - 5)
        rect_end = (x1 + text_width + 5, y1)
        cv2.rectangle(frame, rect_start, rect_end, detection_color, -1)
        cv2.putText(frame, label, (int(x1) + 2, int(y1) - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 3, white, 6)
        cv2.rectangle(frame, (5, 5), (1030, 150), offwhite, -1)
        cv2.putText(frame, f'Coffee Packages: {coffee_package_counting}',
                    (6, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, black, 6)

    out.write(frame)
    cv2.imshow('coffe package', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f'video saved to {out_path}')
