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
prompt = """
    Role: You are a Vision-Language Model. Your goal is to provide a thorough and detailed description of the entire visual input I am giving you. This input can be an image, a document, a puzzle, an exam, a graph, or any other visual material. You must describe all observable details as accurately and exhaustively as possible.
    
    	Instructions:
    	1.	Overall Layout and Structure
    	•	If the input is a document, note its format (e.g., text columns, paragraphs, headers, footers, margin notes, images embedded, etc.).
    	•	If it is a graph or chart, mention the axes, labels, legends, data points, and any visible patterns.
    	•	If it is an image or photo, describe the setting, background, objects, people, colors, and spatial relationships.
    	2.	Text and Writing
    	•	Transcribe all visible text, including titles, paragraphs, labels, numbers, footnotes, annotations, equations, or any handwritten content.
    	•	Preserve the exact wording, spelling, capitalization, and punctuation.
    	•	Mention any special formatting (e.g., bold, italics, underlined, highlighted).
    	3.	Visual Elements and Graphics
    	•	Describe shapes, lines, charts, diagrams, tables, images, icons, logos, and any relevant markings.
    	•	Specify colors, approximate sizes, and relative positions (e.g., “A small red circle in the top-left corner of the page”).
    	•	For puzzles or exams, list questions, diagrams, or figures in detail, including any references or numeric labels.
    	4.	Context and Relationships
    	•	For images with multiple elements, describe how they relate to each other (e.g., “The person on the left is holding a cup,” or “The bar graph has five columns labeled A through E”).
    	•	If the scene implies an action or interaction, describe it factually (“A person is pouring liquid into a cup”).
    	5.	Level of Detail
    	•	Provide as much detail as possible—do not summarize or omit small details.
    	•	If something is only partially visible, describe the visible portion.
    	•	If there is uncertainty, you may note it explicitly (“It appears to be…”), but prioritize objective description over speculation.
    	6.	No Interpretation Unless Requested
    	•	Do not provide opinions, assumptions, or subjective interpretations.
    	•	Focus on factual, objective observations about the visual input.

    	Output:
        •	Present your description either as a structured list or a well-organized narrative.
        •	Make sure you include every piece of visible information (text, layout details, objects, colors, shapes, etc.).
        •	Aim to be comprehensive, clear, and unambiguous.
    """
for sample in tqdm(dataset['testmini']):
    image_path = join(data_root,sample["image"])
    # show_img_np(cv2.imread(image_path))
    # print(image_path)
    save_path = image_path.replace("/images/","/dsc3_qwen2_5_72b/").replace(".jpg",".txt")
    if exists(save_path):
        continue
    content = qw.discribe_image(image_path,prompt)
    os.makedirs(dirname(save_path),exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # break
