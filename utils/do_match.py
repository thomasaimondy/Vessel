'''
视频帧匹配脚本
'''
import numpy as np
import cv2

#至少10个点匹配
MIN_MATCH_COUNT = 10
#完全匹配偏移 d<4
BEST_DISTANCE = 4
#微量偏移  4<d<10
GOOD_DISTANCE = 10


# 特征点提取方法，内置很多种
algorithms_all = {
    "SIFT": cv2.xfeatures2d.SIFT_create(),
    "SURF": cv2.xfeatures2d.SURF_create(8000),
    "ORB": cv2.ORB_create()
}

'''
# 图像匹配
# 0完全不匹配 1场景匹配 2角度轻微偏移 3完全匹配
'''
def match2frames(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    size1 = img1.shape
    size2 = img2.shape

    img1 = cv2.resize(img1, (int(size1[1]*0.3), int(size1[0]*0.3)), cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, (int(size2[1]*0.3), int(size2[0]*0.3)), cv2.INTER_LINEAR)

    sift = algorithms_all["SIFT"]

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # 过滤
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) <= MIN_MATCH_COUNT:
        return 0  # 完全不匹配
    else:
        distance_sum = 0  # 特征点2d物理坐标偏移总和
        for m in good:
            distance_sum += get_distance(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt)
        distance = distance_sum / len(good)  #单个特征点2D物理位置平均偏移量

        if distance < BEST_DISTANCE:
            return 3  #完全匹配
        elif distance < GOOD_DISTANCE and distance >= BEST_DISTANCE:
            return 2  #部分偏移
        else:
            return 1  #场景匹配


'''
计算2D物理距离
'''
def get_distance(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)



if __name__ == "__main__":
    pass