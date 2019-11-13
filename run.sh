gst-launch-1.0 -q v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw, format=BGR ! fdsink | ./demo.py | ./draw.py
