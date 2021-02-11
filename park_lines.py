import cv2


image = cv2.imread(r"PKLot/PKLot/UFPR05/Sunny/2013-02-24/2013-02-24_06_15_00.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blur_gray = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)
edged = cv2.Canny(blur_gray, 30, 200)
cv2.imshow("edged", edged)
cv2.waitKey(0)

contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(contours)
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)

    if len(approx) == 2 or len(approx) == 3:
        screenCnt = approx
        cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow("image", image)
cv2.waitKey(0)