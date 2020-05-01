## Pipeline
![Wait](https://pyimagesearch.com/wp-content/uploads/2018/09/opencv_ocr_pipeline.png)

## Usage
- ``python text_recognition.py --east frozen_east_text_detection.pb --image example.jpg --padding 0.05``

## Issues
- The text detector is not able to predict bounding box in marksheet but is able to do it in any other scans like Highway boards,
books,etc. The reason is scanned images background has watermarks also the text is in block-letter fonts.
