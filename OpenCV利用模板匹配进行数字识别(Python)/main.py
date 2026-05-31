import cv2
import numpy as np
import imutils
import argparse
from imutils import contours
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
img = cv2.imread(r'img/模板.jpg') # 读取模板图片
cv_show('img',img)
ref=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 将图片改为灰度图
cv_show('ref',ref)
ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # 进行二值化处理,cv2.THRESH_BINARY表示超出阈值则设为maxval，cv2.THRESH_BINARY_INV相反
# 这样就可以实现黑白互换
cv_show('ref',ref)
# 现在来找轮廓(只找外轮廓)
refcnts,hierarchy=cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # 用ref.copy()的原因是findContours会改变输入图像
# cv2.RETR_EXTERNAL:只找外轮廓  cv2.CHAIN_APPROX_SIMPLE:只保留终点坐标  cv2.RETR_TREE:内外轮廓都画
# refcnts:轮廓列表   hierarchy:轮廓层级
# 现在进行轮廓绘图
cv2.drawContours(img,refcnts,-1,(0,0,255),3) # 第三个参数填-1表示绘制所有轮廓
cv_show('img',img)
print(f'轮廓个数为:{len(refcnts)}')
refcnts,boxs=imutils.contours.sort_contours(refcnts, method="left-to-right") # 顺序为从左到右，从上到下
# 第一个返回值为排序后的轮廓图像，第二个返回值为每个轮廓的外接矩形的坐标
digit={} # 用于存储轮廓
for (i,c) in enumerate(refcnts): # enumerate 就是 “边遍历，边自动给你编号”，同时拿到 索引 + 元素。
    x,y,w,h=cv2.boundingRect(c) # 给一个轮廓，自动算出能包裹它的最小正矩形（水平垂直，不旋转），返回 (x, y, w, h) 四个值。
    roi=ref[y:y+h,x:x+w]
    roi=cv2.resize(roi,(57,88)) # 把矩形画大点
    digit[i]=roi
# 接下来我们对输入数据进行一些处理
# 初始化卷积核
rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3)) # ksize是卷积核的大小
# cv2.MORPH_RECT:整个小方块全是实心的(矩形核) cv2.MORPH_CROSS:只有中间十字是实心，四角是空的(十字核) cv2.MORPH_ELLIPSE:接近椭圆 / 圆形的实心(椭圆核)
sqKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# 定义一个较小的核，专门去噪点
noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# 读取输入数据，进行预处理
input=cv2.imread(r'img/图片.jpg')
input = imutils.resize(input, width=600) # 放大图片，适配小数字
cv_show('input',input)
gray=cv2.cvtColor(input,cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)
# 进行礼帽操作，突然更明亮的区域
tophat=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel) # 礼帽=原始输入-开运算
cv_show('tophat',tophat)
gradX=cv2.Sobel(tophat, cv2.CV_64F, 1, 0, ksize=-1)
gradX=cv2.convertScaleAbs(gradX) # Sobel算子:图像边缘检测，计算图像水平 / 垂直方向的梯度（明暗变化）
gradX = cv2.normalize(gradX, None, 0, 255, cv2.NORM_MINMAX) # 进行归一化操作
cv_show('gradX',gradX)
# 现在我们采用闭运算让数字能够连在一起
gradX=cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)
cv_show('gradX',gradX)
# THRESH_OTSU会自动找到合适的阈值，适合双峰，需要把阈值参数设置为0
thresh=cv2.threshold(gradX,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # 这里写[1]是因为[0]返回的是计算出的阈值，[1]才返回的是图像
cv_show('thresh',thresh)
# 再来一个闭运算来进行空缺填补
thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)
cv_show('thresh',thresh)
# 现在进行轮阔绘制
threshcnts,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=threshcnts
cur_img=input.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show('cur_img',cur_img)
locs=[] # 用于存放合适的轮廓
# 现在进行轮廓遍历
for i,c in enumerate(cnts):
    dx,dy,dw,dh=cv2.boundingRect(c)
    ar=dw/float(dh) # 这一步是求轮廓的宽高比，以此来过滤一些轮廓
    # 选择合适的区域，根据实际任务来
    if ar>2.5 and ar<5.0:
        if(dw>65 and dw<180) and (dh>15 and dh<40):
            locs.append((dx,dy,dw,dh))
# 将符合的轮廓从左到右排序
locs=sorted(locs,key=lambda x:x[0])
output=[]
# 遍历每个轮廓中的数字
for (i,(zx,zy,zw,zh)) in enumerate(locs):
    groupoutput=[]
    group=gray[zy-5:zy+zh+5,zx-5:zx+zw+5]
    cv_show("group",group)
    # 再次进行预处理
    group=cv2.GaussianBlur(group, (3,3), 0) # 去噪
    group=cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show("group",group)
    # 依旧计算轮廓
    digitcnts,hierarchy=cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 进行排序
    digitcnts,_=contours.sort_contours(digitcnts, method="left-to-right")
    # 计算数值
    for c in digitcnts:
        mx,my,mw,mh=cv2.boundingRect(c)
        roi=group[my:my+mh,mx:mx+mw]
        roi=cv2.resize(roi,(57,88))
        cv_show('roi',roi)
        scores=[]
        for (digit_num,digitROI) in digit.items():
            result=cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF_NORMED)
            (_,score,_,_)=cv2.minMaxLoc(result)
            scores.append(score)
        groupoutput.append(str(np.argmax(scores)))
        # 画出来
        cv2.rectangle(input, (zx - 5, zy - 5), (zx + zw + 5, zy + zh + 5), (0, 255, 0), 2)
        cv2.putText(input, "".join(groupoutput), (zx, zy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        output.extend(groupoutput)
cv_show('input',input)