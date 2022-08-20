# OpenCV_Hazit
This project was created by Roi Tiefenbrunn and Zon Ziskind as part of the "Research Front in Computer Science" By Ben-Gurion university
In this project, led by Jihad El-Sana, we attempted to create an image processing algorithm which finds a quadrilateral in a given image. 
More information about the approach and conclusions can be found in the report.

manuals for the compiled distributables which reside at the dist directory:

# frame_calculator.exe (the whole algorithm):

synopsis:
frame_calculator.exe --image path/to/image.png [--mask path/to/mask.png] [--color red|green|blue]

description:
-i, --image path/to/image.png
    image input to process
-m, --mask
    mask input, not mandatory
-c, --color
    in what color should the final lines be drawn with (red, green, or blue), not mandatory
    
    
# Lines_Stage.exe (the line finding process - after hough space was calculated)

synopsis:
Lines_Stage.exe --image path/to/image.png --grad path/to/gradient.txt --space path/to/hough_space.txt [--color red|green|blue]

description:
--image path/to/image.png
    image input to process
--grad path/to/gradient.txt
    gradient input
--space path/to/hough_space.txt
    hough-space input
-c, --color
    in what color should the final lines be drawn with (red, green, or blue), not mandatory
