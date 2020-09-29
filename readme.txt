looks like these are some good libraries:
Keras
OpenCV

basically gonna copy this:
www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras

interesting note:
In python, this

	img[range(100, 150), range(100, 150)] = [0,0,0]

creates a black line on the image connecting (100, 100) to (150, 150).
It does not end up filling in the square between those points; it
simply creates a line with slope -1.

Note: Not going to download the data from the website; im just gonna make
something myself.
