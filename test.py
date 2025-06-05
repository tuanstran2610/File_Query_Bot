from PIL import Image
import cv2

img = cv2.imread("./Upload/ChuaNhan.png")
print("Original shape:", img.shape)

# Save grayscale and thresholded versions
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Image.fromarray(gray).save("gray_output.png")

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
Image.fromarray(thresh).save("thresh_output.png")
