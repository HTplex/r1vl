import matplotlib.pyplot as plt
def show_img_np(img, max_h=3, max_w=20, save="", cmap='gray'):
    """
    :param np_array: input image, one channel or 3 channel,
    :param save: if save image
    :param size:
    :return:
    """
    if len(img.shape) < 3:
        plt.rcParams['image.cmap'] = cmap
    plt.figure(figsize=(max_w, max_h), facecolor='w', edgecolor='k')
    plt.imshow(img)
    if save:
        cv2.imwrite(save, img)
    else:
        plt.show()

# data explore
import cv2
from os.path import join, dirname, exists
from qwen2vl_worker import QwenVLWorker
import os 


data_root = "/home/agent_h/data/datasets/MathVision"
from datasets import load_dataset
qw = QwenVLWorker()

from tqdm import tqdm
dataset = load_dataset(data_root)
for sample in tqdm(dataset['testmini']):
    image_path = join(data_root,sample["image"])
    # show_img_np(cv2.imread(image_path))
    # print(image_path)
    save_path = image_path.replace("/images/","/dsc1_qwen2_72b/").replace(".jpg",".txt")
    if exists(save_path):
        continue
    content = qw.discribe_image(image_path)
    os.makedirs(dirname(save_path),exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # break
