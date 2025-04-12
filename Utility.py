import os
import cv2
import numpy as np
import matplotlib as plt

img_size = (224, 224)
red_color = (0, 0, 255)
thickness = 1
n_patch_img = 196   # number of patches per image   (196 = 14 x 14)
n_patch_side = np.sqrt(n_patch_img)                     # number of patches per side    (14 = √196)
w_patch = int(img_size[0] / n_patch_side)               # width of patches in pixel     (16px = 224 / 14)
h_patch = int(img_size[1] / n_patch_side)               # height of patches in pixel    (16px = 224 / 14)

def plot_gridded_image(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, img_size)

    for idx in range(n_patch_side):
        
        # vertical lines
        temp = cv2.line(img_resized, (idx * w_patch, 0), (idx * w_patch, img_size[1]), color=red_color, thickness=thickness)
        
        # horizzonal lines
        temp = cv2.line(img_resized, (0, idx * h_patch), (img_size[0], idx * h_patch), color=red_color, thickness=thickness)

    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    plt.imshow(temp)

def get_box_coordinates(idx):
    row, col = int(idx % n_patch_side), int(idx // n_patch_side)

    top_left = (row * h_patch, col * w_patch)
    bottom_right = (row * h_patch + h_patch, col * w_patch + w_patch)

    return row, col, top_left, bottom_right

def get_plot_images(idx, path):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, img_size)
    
    _, _, top_left, bottom_right = get_box_coordinates(idx)
    x, y = top_left[1], top_left[0]

    img_rect = cv2.rectangle(img_resized.copy(), top_left, bottom_right, color=red_color, thickness=thickness)
    img_crop = img_resized.copy()[x : x + h_patch, y : y + w_patch]
    
    return img_rect, img_crop

def show(input_idx, input_path, model, save = False):
    n_patch_img = model.memory_bank.shape[0] // len(model.memory_bank_paths)

    memory_bank_idx = model.dist_score_idxs[model.s_idx]
    target_path_idx = memory_bank_idx // n_patch_img
    target_idx = memory_bank_idx % n_patch_img
    target_path = model.memory_bank_paths[target_path_idx]

    input_rect, input_crop = get_plot_images(input_idx, input_path)
    target_rect, target_crop = get_plot_images(target_idx, target_path)

    # 2x2 plot
    fig, axs = plt.subplots(2, 3, figsize=(8, 8))
    
    fig.suptitle('Patch Analysis', fontsize=14, fontweight='bold')

    axs[0,0].set_title('Input image') # magari mettere anche il nome .jpg
    axs[0,0].imshow(cv2.cvtColor(input_rect, cv2.COLOR_BGR2RGB))

    axs[0,1].set_title('Memory bank image') # magari mettere anche il nome .jpg
    axs[0,1].imshow(cv2.cvtColor(target_rect, cv2.COLOR_BGR2RGB))

    axs[1,0].set_title(f'Patch: {input_idx}, th:0.5') # magari mettere anche il nome .jpg
    axs[1,0].imshow(cv2.cvtColor(input_crop, cv2.COLOR_BGR2RGB))

    axs[1,1].set_title(f'Score: 0.25, result: OK') # magari mettere anche il nome .jpg
    axs[1,1].imshow(cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB))

    axs[0,2].axis('off') # Removes axis form the plot
    axs[1,2].axis('off') # Removes axis form the plot
    
    axs[0,2].text(0, 0.5,
        "Legenda o info extra\n- Blue: dataset A\n- Rosso: dataset B",
        ha='left', va='center', fontsize=10,
        bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10}
    )

    if save:
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        plt.savefig(f'tmp/{os.path.basename(input_path)}_{input_idx}.png')
    else:
        plt.subplots_adjust(top=0.93, bottom=0.4, wspace=0.2)
        plt.show()

def create_gif(input_path, model, n_patch_img):
    
    s, segm_map = model.predict()
    for idx in range(n_patch_img):
        show(idx, input_path, model, save=True)
    
    frames = os.listdir('tmp')
    frames.sort()
    frames[0].save("patch_analysis.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)

