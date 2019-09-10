import cv2
import numpy as np
import imageio
from PIL import Image

points = []
def get_point(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # cv2.circle(img,(x,y),100,(255,0,0),-1)
        points.append([x,y])

img = cv2.imread('even smaller grocery store.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
# img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',get_point)

while(1):
    cv2.imshow('image',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    break

gif = imageio.mimread("delaware.gif")
duration = Image.open("delaware.gif").info['duration']
# print(len(gif)/Image.open("delaware.gif").info['duration'])

nums = len(gif)

y_min, x_min = gif[0].shape[0]*gif[0].shape[1], gif[0].shape[0]*gif[0].shape[1]
y_max, x_max = 0, 0

for point in points:
	if point[0] < x_min:
		x_min = point[0]
	if point[0] > x_max:
		x_max = point[0]
	if point[1] < y_min:
		y_min = point[1]
	if point[1] > y_max:
		y_max = point[1]

print(points)
x_ratio = gif[0].shape[1]/float(x_max-x_min)
y_ratio = gif[0].shape[0]/float(y_max-y_min)

for point in points:
	point[0] = int((point[0]-x_min)*x_ratio)+1
	point[1] = int((point[1]-y_min)*y_ratio)+1
# print(points)
# exit()
# cv2.rectangle(img, (x_min, y_min), \
#           		(x_max, y_max), 2, 2)

# cv2.imshow('image',img)
# cv2.waitKey(0)

pts1 = np.float32(
    [[5, gif[0].shape[0]-6],
     [gif[0].shape[1]-6, gif[0].shape[0]-6],
     [5, 5],
     [gif[0].shape[1]-6, 5]]
)
pts2 = np.float32(
    [[points[2][0], points[2][1]],
     [points[3][0], points[3][1]],
     [points[0][0], points[0][1]],
     [points[1][0], points[1][1]]]
)    
M = cv2.getPerspectiveTransform(pts1,pts2)

x_offset = x_min 
y_offset = y_min

# new_img = img

b_channel, g_channel, r_channel = cv2.split(img)
alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.
new_img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
# still_frame = Image.fromarray(new_img)

new_gif = []
for image_itr in range(nums):
	image = gif[image_itr]
	print(image_itr)
	if image_itr == 30:
		break
	dst = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
	dst = cv2.resize(dst, None,fx=1.0/x_ratio, fy=1.0/y_ratio, interpolation = cv2.INTER_CUBIC)

	
	

	# new_img = np.full((img.shape[0],img.shape[1],img.shape[2]+1),0.5)
	# new_img[:,:,:-1] = img
	# dst_no_alpha = np.delete(dst, -1, 2)

	
	s_img = dst

	y1, y2 = y_offset, y_offset + s_img.shape[0]
	x1, x2 = x_offset, x_offset + s_img.shape[1]

	alpha_s = s_img[:, :, 3] / 255.0
	alpha_l = 1.0 - alpha_s

	for c in range(0, 3):
	    new_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
	                              alpha_l * new_img[y1:y2, x1:x2, c])

	# backgroud = Image.fromarray(new_img)
	# front = Image.fromarray(dst)
	# backgroud.paste(front, (x_offset,y_offset))

	# cv2.imshow('image',new_img)
	# cv2.waitKey(0)
	# exit()
	# new_gif.append(new_img)
	new_gif.append(Image.fromarray(new_img).copy())
	# new_gif.append(new_img)

# imageio.mimsave('moving_ball.gif', new_gif, format='GIF', duration=duration)
new_gif[0].save('moving_ball.gif', format='GIF', append_images=new_gif[1:], save_all=True, duration=duration, loop=0)




