import cv2
import numpy as np
import imageio
from PIL import Image
from os import listdir
from os.path import isfile, join
import pickle
import sys
import urllib
import urllib.request
from urllib.request import urlopen




def make_over_lay(backgroud_images, gif, nums, bounds):
    for item in backgroud_images:
        backgroud = item[0]
        name = item[1]

        y_min, x_min = gif[0].shape[0]*gif[0].shape[1], gif[0].shape[0]*gif[0].shape[1]
        y_max, x_max = 0, 0
        points = bounds # overlay_loc[name]
        for point in points:
            if point[0] < x_min:
                x_min = point[0]
            if point[0] > x_max:
                x_max = point[0]
            if point[1] < y_min:
                y_min = point[1]
            if point[1] > y_max:
                y_max = point[1]

        x_ratio = gif[0].shape[1]/float(x_max-x_min)
        y_ratio = gif[0].shape[0]/float(y_max-y_min)

        for point in points:
            point[0] = int((point[0]-x_min)*x_ratio)+1
            point[1] = int((point[1]-y_min)*y_ratio)+1


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


        b_channel, g_channel, r_channel = cv2.split(backgroud)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.
        backgroud = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        # still_frame = Image.fromarray(new_img)


        new_gif = []
        for image_itr in range(nums):
            image = gif[image_itr]
            # if image_itr == 30:
            # 	break
            dst = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
            dst = cv2.resize(dst, None,fx=1.0/x_ratio, fy=1.0/y_ratio, interpolation = cv2.INTER_CUBIC)


            
            s_img = dst
            # print ("here is s_img", s_img)

            y1, y2 = y_offset, y_offset + s_img.shape[0]
            x1, x2 = x_offset, x_offset + s_img.shape[1]

            alpha_s = s_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                backgroud[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                        alpha_l * backgroud[y1:y2, x1:x2, c])


            new_gif.append(Image.fromarray(backgroud).copy())
            # new_gif.append(new_img)

        # imageio.mimsave('moving_ball.gif', new_gif, format='GIF', duration=duration)
        new_gif[0].save('Processed GIFs/'+name.split(".")[0]+'.gif', format='GIF', append_images=new_gif[1:], save_all=True, duration=duration, loop=0)
        print(name.split(".")[0]+'.gif - saved in Processed GIFs')


if __name__ == "__main__":
    
    

    background_url = "https://geoswap-customer-assets.s3.amazonaws.com/goldfish/images/Bars_Cropped.jpeg"
    gif_url = "https://geoswap-customer-assets.s3.amazonaws.com/goldfish/images/delaware.gif"
    bounds = [[102, 122], [305, 117], [107, 249], [319, 231]]
    
    resp = urllib.request.urlopen(background_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
   
    gif = imageio.mimread(imageio.core.urlopen(gif_url).read(), '.gif')
    
    print ("frames" , len(gif))


    nums = len(gif)
    duration = 40 #this means there are 25 frames per second
    
    make_over_lay([[img,'Bars_Cropped3Urldoulbe_17ee1D.jpeg']], gif, nums, bounds )


    