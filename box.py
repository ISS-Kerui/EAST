import cv2
import numpy as np

def get_sub_imgs(img_path, text_path, index):
	img = cv2.imread(img_path)
	points = []
	with open(text_path) as f:
		for line in f.readlines():
			line = line.strip().split(',')
			for i in range(0,8,2):
				points.append((int(line[i]),int(line[i+1])))
	points = np.array(points)
	for i in range(0,len(points),4):
		rect = cv2.minAreaRect(points[i:i+4])
		box = cv2.boxPoints(rect) 
		box = np.int0(box)

		#cv2.drawContours(img, [box], 0, (0, 0, 255), 2) 
		x, y, w, h = cv2.boundingRect(box)
		cut_img = img[y:y+h, x:x+w]
		cv2.imwrite('text_'+str(index)+'.jpg', cut_img)
		index += 1
		#
	# for point in points:
	# 	cv2.circle(img, point, 1, (255, 0, 0), thickness = -1)
	# cv2.namedWindow("Image")
	# cv2.imshow("image", img)
	# cv2.waitKey(0)

if __name__ == '__main__':
	img_path = 'img_2.jpg'
	text_path = 'demo_output/img_2.txt'
	get_sub_imgs(img_path, text_path,0)