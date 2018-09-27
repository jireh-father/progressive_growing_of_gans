import cv2
import os, glob

img_dir = "/home/data4/dataset/cate_cut_0411/shirt"
output_dir = img_dir + "_edge"
assert img_dir
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
imgs = glob.glob(os.path.join(img_dir, "*.jpg"))
# plt.subplot(121)
for i, path in enumerate(imgs):
    img = cv2.imread(path, 0)
    edges = cv2.Canny(img, 20, 70)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(path)), edges, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
