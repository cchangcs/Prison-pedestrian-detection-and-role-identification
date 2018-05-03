# encoding:utf-8
import os
import numpy as np
import torch
from crowd_count import CrowdCounter
import cv2
import network
import sys


# 作用：用于得到图像的边缘
# img:输入图像
# width,height:图像的宽度，用做图像的归一化
def gradient(img, width, height):
    img = cv2.imread(img, 0)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 对x求导
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 对y求导
    abs_x = cv2.convertScaleAbs(x)  # 转回uint8
    abs_y = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    return dst.astype(np.uint8, copy=False)


# 作用：用于得到预测结果
# per_img:目标图片
# back_img：背景图
# threshold：阈值
# model_path：模型目录
# scene:场景标识（1：走廊 2：会见室）
def detect(per_img, back_img, threshold, model_path, scene):

    # 用于存储最终结果
    result_dic = {}
    person_list = []
    total_person = 0
    police_num = 0
    prisoner_num = 0
    other = 0

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    vis = False
    save_output = True

    model_name = os.path.basename(model_path).split('.')[0]
    # 初始化网络
    net = CrowdCounter()

    trained_model = os.path.join(model_path)
    # 加载权值文件
    network.load_net(trained_model, net)
    # net.cuda() 用于进行cuda加速
    net.eval()
    # 读取图片
    img_r = cv2.imread(per_img)
    img1 = cv2.imread(per_img, 0)
    # 保存灰度化之后的图片
    # img_gray = img1
    img = img1.astype(np.float32, copy=False)
    ht = img.shape[0]
    wd = img.shape[1]
    ht_1 = int((ht / 4) * 4)
    wd_1 = int((wd / 4) * 4)
    img = cv2.resize(img, (wd_1, ht_1))
    img = img.reshape((1, 1, img.shape[0], img.shape[1]))

    # 对图片进行预测
    heat_map = net.forward(img)
    heat_map = torch.squeeze(heat_map)
    heat_map = heat_map.data.numpy()
    [height, width] = heat_map.shape
    dst = cv2.resize(img_r, (width, height), interpolation=cv2.INTER_LINEAR)

    # img_gray = cv2.resize(img_gray, (width, height), interpolation=cv2.INTER_LINEAR)
    # 计算背景图和目标图片的梯度
    hist1 = gradient(per_img, width, height)
    hist2 = gradient(back_img, width, height)

    result = np.empty([height, width])
    # 计算梯度差
    for i in range(width - 1):
        for j in range(height - 1):
            result[j, i] = abs(hist1[j, i] - hist2[j, i])
    # 输出提督差结果
    # for i in range(width - 1):
    #     for j in range(height - 1):
    #         print('{0}-{1},{2}'.format(result[j, i], i, j))
    # 根据梯度差得出目标区域
    l = []
    for i in range(width - 1):
        for j in range(height - 1):
            # print('{0}-({1},{2})'.format(result[j, i], j, i))
            count = 0
            if heat_map[j, i] > 0:  # 预测结果大于0即可能是人
                # dst = cv2.circle(dst, (i, j), 1, (0, 0, 255), -1)
                # 将目标点向左向右个扩充15个像素向下扩充50个像素对比梯度差
                for ix in range(30):
                    for jy in range(50):
                        if (jy + j > 0) and (jy + j < height - 1) and (ix + i - 5 > 0) and (ix + i - 5 < width - 1):
                            if result[jy + j, ix + i - 15] > threshold:
                                # 输出满足条件的点的阈值和坐标
                                count += 1
                # 根据图片的大小对左下角和右上角文字区域进行重点过滤
                if height == 180 and width == 320:
                    if j > 22 and (i < 175 or j < 156) and (i < 276 or j < 127):
                        if count > 500:
                            l.append([i, j])
                            # 在满足条件的地方绘制十字
                            # dst = cv2.circle(dst, (i, j), 1, (0, 0, 255), -1)
                elif height == 144 and width == 176:
                    if (i > 87 or j > 12) and (i < 116 or j < 122) and (i < 148 or j < 103):
                        if count > 500:
                            l.append([i, j])
                            # dst = cv2.circle(dst, (i, j), 1, (0, 0, 255), -1)
                else:
                    if j > 22 and (i < 175 or j < 156) and (i < 276 or j < 127):
                        if count > 500:
                            l.append([i, j])
                            # 在满足条件的地方绘制十字
                            # dst = cv2.circle(dst, (i, j), 1, (0, 0, 255), -1)

    rects = []
    entry = True
    x = 0
    y = 0
    old_x = 0
    old_y = 0
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lower_blue = np.array([110, 100, 50])
    upper_blue = np.array([130, 255, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([170, 255, 48])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_black, upper_black)
    # 将结果与原图做按位与
    # blueThings = cv2.bitwise_and(dst, dst, mask=mask)

    # 对目标区域通过矩形框进行聚类
    # 走廊/ab门
    if scene == 1:
        for (i, j) in l:
            if entry:
                # cv2.rectangle(dst, (i - 10, j - 10), (i + 20, j + 40), (0, 255, 0), 1)
                rects.append([i - 10, j - 10, 30, 35])
                x = i + 10
                y = j + 20
                old_x = i
                old_y = j
                entry = False
            if i > x or (j - y) > 35 or (old_y - j > 15 and abs(old_x - i) < 15):
                entry = True
            # else:
            #     if j < old_y:
            #         rects.pop()
            #         rects.append([i - 10, j - 10, 30, 35])

        # 对图片进行二值化
        # ret, thresh1 = cv2.threshold(img_gray, 75, 255, cv2.THRESH_BINARY)
        # cv2.imshow("binary", thresh1)
        # thresh1 = thresh1.astype(np.uint32, copy=False)  # for i in range(width - 1):
        #     for j in range(height - 1):
        #         print('{0}-({1},{2})'.format(thresh1[j, i], i, j))
        # 对矩形框区域使用梯度差进行再次过滤
        l2 = []
        # 通过颜色判断人物的角色
        # 黑色：警察
        # 蓝色：囚犯
        # 其他：普通人
        for (x, y, w, h) in rects:
            count = 0
            # 保存矩形框类的黑色和蓝色区域
            black_count = 0
            blue_count = 0
            # 用于标示是否已添加文字类别
            label_entry = True
            for i in range(w - 1):
                for j in range(h - 1):
                    # 人员信息
                    person_dic = {}
                    if mask1[y + j, x + i] == 255:
                        blue_count += 1
                    if mask2[y + j, x + i] == 255:
                        black_count += 1
                    # if thresh1[y + j, x + i] == 255:
                    #     thresh_count = thresh_count + 1
                    if result[y + j, x + i] > threshold:
                        count += 1
                    if count > 350:  # and thresh_count < (w * h)/3:
                        l2.append([x, y, w, h])
                        # 根据颜色阈值对警察和囚犯进行标记
                        if label_entry:
                            if blue_count > 50:
                                total_person += 1  # 用于计数总人数
                                prisoner_num += 1  # 用于计数囚犯数目
                                label_entry = False
                                person_dic['position'] = [x, y, x+w, y+h]
                                person_dic['role'] = 'prisoner'
                                cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 1)
                                cv2.putText(dst, 'prisoner', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                                person_list.append(person_dic)
                            elif black_count > 170:
                                total_person += 1  # 用于计数总人数
                                police_num += 1  # 用于计数警察数目
                                label_entry = False
                                person_dic['position'] = [x, y, x + w, y + h]
                                person_dic['role'] = 'police'
                                cv2.rectangle(dst, (x, y), (x + w, y + h), (255, 0, 0), 1)
                                cv2.putText(dst, 'police', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                                person_list.append(person_dic)
            # 人员信息
            person_dic = {}
            if count > 350 and label_entry:
                label_entry = False
                total_person += 1  # 用于计数总人数
                other += 1  # 用于计数普通人数目
                person_dic['position'] = [x, y, x + w, y + h]
                person_dic['role'] = 'other'
                cv2.rectangle(dst, (x, y), (x + w, y + h), (255, 255, 0), 1)
                cv2.putText(dst, 'other', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                person_list.append(person_dic)

    # 会见室
    elif scene == 2:
        l.reverse()
        for (i, j) in l:
            if entry:
                # cv2.rectangle(dst, (i - 10, j - 10), (i + 20, j + 25), (0, 255, 0), 1)
                rects.append([i - 10, j - 10, 30, 35])
                x = i + 20
                y = j + 20
                entry = False
            if i > x or j > y:
                entry = True
        # 对矩形框区域使用梯度差进行再次过滤
        l2 = []
        for (x, y, w, h) in rects:
            count = 0
            # 保存矩形框类的黑色和蓝色区域
            black_count = 0
            blue_count = 0
            # 用于标示是否已添加文字类别
            label_entry = True
            for i in range(w - 1):
                for j in range(h - 1):
                    # 人员信息
                    person_dic = {}
                    if mask1[y + j, x + i] == 255:
                        blue_count += 1
                    if mask2[y + j, x + i] == 255:
                        black_count += 1
                    if result[y + j, x + i] > threshold:
                        count += 1
                    if count > 420:
                        l2.append([x, y, w, h])
                        if label_entry:
                            if blue_count > 50:
                                total_person += 1  # 用于计数总人数
                                prisoner_num += 1  # 用于计数囚犯数目
                                label_entry = False
                                person_dic['position'] = [x, y, x + w, y + h]
                                person_dic['role'] = 'prisoner'
                                cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 1)
                                cv2.putText(dst, 'prisoner', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                                person_list.append(person_dic)
                            elif black_count > 170:
                                label_entry = False
                                total_person += 1  # 用于计数总人数
                                police_num += 1  # 用于计数警察数目
                                person_dic['position'] = [x, y, x + w, y + h]
                                person_dic['role'] = 'police'
                                cv2.rectangle(dst, (x, y), (x + w, y + h), (255, 0, 0), 1)
                                cv2.putText(dst, 'police', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                                person_list.append(person_dic)
            # 人员信息
            person_dic = {}
            if count > 420 and label_entry:
                label_entry = False
                total_person += 1  # 用于计数总人数
                other += 1  # 用于计数普通人数目
                person_dic['position'] = [x, y, x + w, y + h]
                person_dic['role'] = 'other'
                cv2.rectangle(dst, (x, y), (x + w, y + h), (255, 255, 0), 1)
                cv2.putText(dst, 'other', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                person_list.append(person_dic)
            print(count)
    # 组装生成最后的结果
    result_dic['total_people'] = total_person
    result_dic['police_num'] = police_num
    result_dic['prisoner_num'] = prisoner_num
    result_dic['other'] = other
    result_dic['people_list'] = person_list
    print(result_dic)
    cv2.namedWindow("result")
    cv2.imshow("result", dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result_dic

if __name__ == '__main__':
    # 通用阈值：120（会见室和ab门阈值）
    detect('test3/people/1.jpg', 'test3/room/1.jpg', 120, 'models/mcnn_shtechB_110.h5', 1)
    # per_img = sys.argv[1]
    # back_img = sys.argv[2]
    # threshold = int(sys.argv[3])
    # scene = int(sys.argv[4])
    # print(sys.argv)
    # detect(per_img, back_img, threshold, 'pyfile/models/mcnn_shtechB_110.h5', scene)
    # 用于确定绿色和黑色的阈值以达到区分出警察和犯人的作用
    # rgb_img = cv2.imread('test3/people/ab5.jpg')
    # rgb_img = cv2.resize(rgb_img, (500, 400), interpolation=cv2.INTER_LINEAR)
    # HSV = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    # H, S, V = cv2.split(HSV)
    # lowerBlue = np.array([110, 100, 50])
    # upperBlue = np.array([130, 255, 255])
    # lowerBlack = np.array([0, 0, 0])
    # upperBlack = np.array([170, 255, 48])
    # mask = cv2.inRange(HSV, lowerBlack, upperBlack)
    # blueThings = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         print(mask[i, j])
    # cv2.imshow('resource', rgb_img)
    # cv2.imshow('result', blueThings)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
