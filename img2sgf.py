# Work in progress: load an image and see if we can find a 19x19 grid
# First attempt: preprocess and plot the lines in hough space:
#   if this works, we should see two rows of equally spaced blobs representing the
#   horizontal and vertical grid lines.
# We get blobs not dots because each line is picked up multiple times
# Next steps:
#   apply a clustering algorithm to get unique lines
#   adaptive thresholding: iterate and find the threshold that gives us most/all grid lines
#   validate that it's a real 19x19 grid; fill in blanks if needed
#   identify intersections at empty/black/white
#   output in SGF format
#   if I don't get bored with this, make a nice GUI so that other people can easily use it

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys, math

input_file = sys.argv[1] # To do: sanity checking of command line arguments
if len(sys.argv)>2:
  threshold = int(sys.argv[2])
else:
  threshold = 80
maxblur = 2
angle_delta = math.pi/180 # accept lines up to 2 degrees away from horizontal or vertical

input_image = cv.imread(input_file)
grey_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
edge_detected_image = cv.Canny(input_image, 50, 200)
circles_removed_image = edge_detected_image.copy()

# Make a few different blurred versions of the image, so we can find most of the circles
blurs = [grey_image]
for i in range(maxblur):
  b = 2*i + 1
  blurs.append(cv.medianBlur(grey_image, b))
  blurs.append(cv.GaussianBlur(grey_image, (b,b), b))

first_circles = True
for b in blurs:
  c = cv.HoughCircles(b, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
  if first_circles:
    circles = c[0]
    first_circles = False
  else:
    circles = np.vstack((circles, c[0]))

# For each circle, erase the bounding box and replace by a single pixel in the middle
for i in range(circles.shape[0]):
  xc, yc, r = circles[i,:]
  r = r+2 # need +2 because circle edges can stick out a little past the bounding box
  ul = (int(round(xc-r)), int(round(yc-r)))
  lr = (int(round(xc+r)), int(round(yc+r)))
  middle = (int(round(xc)), int(round(yc)))
  cv.rectangle(circles_removed_image, ul, lr, (0,0,0), -1)  # -1 = filled
  cv.circle(circles_removed_image, middle, 1, (255,255,255), -1)

hlines = cv.HoughLines(circles_removed_image, rho=1, theta=math.pi/180.0, threshold=threshold, \
          min_theta = math.pi/2 - angle_delta, max_theta = math.pi/2 + angle_delta)
if hlines is None:
  sys.exit("Error: no horizontal lines found")
vlines1 = cv.HoughLines(circles_removed_image, rho=1, theta=math.pi/180.0, threshold=threshold, \
          min_theta = 0, max_theta = angle_delta)
vlines2 = cv.HoughLines(circles_removed_image, rho=1, theta=math.pi/180.0, threshold=threshold, \
          min_theta = math.pi - angle_delta, max_theta = math.pi)
if vlines1 is None and vlines2 in None:
  sys.exit("Error: no vertical lines found")
if vlines2 is not None:
  vlines2[:,0,0] = -vlines2[:,0,0]
  vlines2[:,0,1] = vlines2[:,0,1] - math.pi
  if vlines1 is not None:
    vlines = np.vstack((vlines1, vlines2))
  else:
    vlines = vlines2
else:
  vlines = vlines1

all_lines = np.vstack((hlines, vlines))
plt.scatter(all_lines[:,0,0], all_lines[:,0,1])
plt.show()