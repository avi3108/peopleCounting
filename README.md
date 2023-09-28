**People Counting**

This is a Python implementation of a person counting algorithm using the YOLOv8 object detection model and a centroid tracker. The algorithm can be used to count the number of people entering and exiting a premise from a given video feed.

**Requirements**

* Python 3
* OpenCV
* NumPy
* Pandas
* Ultralytics
* Tracker

**Installation**

```
pip install -r requirements.txt
```

**Usage**

To run the people counter, simply execute the following command:

```
python3 mains.py
```

The program will read the input video file from the `input_videos` directory and write the output video file to the `output_videos` directory. The output video file will display the entry and exit counts on the video screen.

**Command-line Options**

* `-i`: Path to the input video file.
* `-o`: Path to the output video file.
* `-f`: Frames per second.
* `-s`: Frame size.

**Example**


python3 mains.py -i input_videos/exit_example.avi -o output_videos/exit_example_output.avi -f 10 -s 1020x500


**Contributing**

Contributions are welcome! Please feel free to fork the repository and submit a pull request with your changes.

**License**

This code is licensed under the MIT License.

**Contact**

If you have any questions or feedback, please feel free to contact me at [email protected]