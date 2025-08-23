import os
import cv2
import json
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

# mvtec_classes = [ "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper" ]
mvtec_classes = [ "bottle", "cable"]
img_size = (224, 224)
red_color = (0, 0, 255)
thickness = 1
n_patch_img = 196   # number of patches per image   (196 = 14 x 14)
n_patch_side = int(np.sqrt(n_patch_img))                # number of patches per side    (14 = √196)
w_patch = int(img_size[0] / n_patch_side)               # width of patches in pixel     (16px = 224 / 14)
h_patch = int(img_size[1] / n_patch_side)               # height of patches in pixel    (16px = 224 / 14)

# UTILITY FUNCTIONS FOR IMAGE ANALYSIS

def plot_gridded_image(img_path: str):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, img_size)

    for idx in range(n_patch_side):
        
        # vertical lines
        temp = cv2.line(img_resized, (idx * w_patch, 0), (idx * w_patch, img_size[1]), color=red_color, thickness=thickness)
        
        # horizzonal lines
        temp = cv2.line(img_resized, (0, idx * h_patch), (img_size[0], idx * h_patch), color=red_color, thickness=thickness)

    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    plt.imshow(temp)
    plt.show()

def get_box_coordinates(idx: int):
    row, col = int(idx % n_patch_side), int(idx // n_patch_side)

    top_left = (row * h_patch, col * w_patch)
    bottom_right = (row * h_patch + h_patch, col * w_patch + w_patch)

    return row, col, top_left, bottom_right

def get_plot_images(idx: int, path: str):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, img_size)
    
    _, _, top_left, bottom_right = get_box_coordinates(idx)
    x, y = top_left[1], top_left[0]

    img_rect = cv2.rectangle(img_resized.copy(), top_left, bottom_right, color=red_color, thickness=thickness)
    img_crop = img_resized.copy()[x : x + h_patch, y : y + w_patch]
    
    return img_rect, img_crop, img_resized

# Given an patch index it creates a plot 
def show(input_idx: int, input_path: str, model, save = False, alpha = 0.7):

    memory_bank_idx = model.dist_score_idxs[input_idx]
    target_path_idx = memory_bank_idx // n_patch_img
    target_idx = memory_bank_idx % n_patch_img
    target_path = model.memory_bank_paths[target_path_idx][0] # TODO: Perché restituisce una tupla

    input_rect, input_crop, input_resized = get_plot_images(input_idx, input_path)
    target_rect, target_crop, _  = get_plot_images(target_idx, target_path)

    # 2x2 plot
    fig, axs = plt.subplots(2, 3, figsize=(8, 8))
    
    fig.suptitle('Patch Analysis', fontsize=14, fontweight='bold')

    axs[0,0].set_title('Input image') 
    input_rect_img = cv2.cvtColor(input_rect, cv2.COLOR_BGR2RGB)
    axs[0,0].imshow(input_rect_img)

    axs[0,1].set_title(f'{os.path.basename(target_path)}') # magari mettere anche il nome .jpg
    target_rect_img = cv2.cvtColor(target_rect, cv2.COLOR_BGR2RGB)
    axs[0,1].imshow(target_rect_img)

    axs[1,0].set_title(f'Patch: {input_idx}') 
    input_crop_img = cv2.cvtColor(input_crop, cv2.COLOR_BGR2RGB)
    axs[1,0].imshow(input_crop_img)

    axs[1,1].set_title(f'Score: {model.score:.2f}') 
    target_crop_img = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
    axs[1,1].imshow(target_crop_img)

    axs[0,2].axis('off') # Removes axis form the plot
    axs[1,2].axis('off') # Removes axis form the plot
    
    axs[0,2].set_title(f'Heat map')
    axs[0,2].imshow(input_resized)
    heat_map = model.segm_map.reshape(img_size)
    axs[0,2].imshow(heat_map, alpha=alpha)
    heatmap_img = cv2.addWeighted(heat_map, alpha, input_resized, 1 - alpha, 0)
    cv2.imwrite("temp.png", heatmap_img)

    axs[1,2].text(0, 0.5,
        "Legend or extra info\n- Blue: dataset A\n- Red: dataset B",
        ha='left', va='center', fontsize=10,
        bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10}
    )

    if save:
        if not os.path.exists('tmp_img'):
            os.makedirs('tmp_img')
        plt.savefig(f'tmp_img/{input_idx:03d}.png')
        plt.close()
    else:
        plt.subplots_adjust(top=0.93, bottom=0.4, wspace=0.2)
        plt.show()

# Creates a gif given a trained model
def create_gif(input_path: str, model, duration:int=100, output_path:str="patch_analysis.gif" ):
    
    n_patch_img = model.memory_bank.shape[0] // len(model.memory_bank_paths)

    img = Image.open(input_path).convert("RGB")
    sample = model.processor(img)
    sample_torch = torch.Tensor(sample['pixel_values'][0]).unsqueeze(0)
    sample['pixel_values'][0] = sample_torch
    
    _, _ = model.predict(sample)

    # For each patch it creates a frame
    [ show(idx, input_path, model, save = True) for idx in tqdm(range(n_patch_img)) ]# Change to save = True
    
    frames_paths = sorted([f for f in os.listdir('tmp') if f.endswith(('.png', '.jpg', '.jpeg'))])
    frames = [Image.open(os.path.join('tmp', f)) for f in frames_paths]
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    
    # shutil.rmtree('tmp_img')

# UTILITY FUNCTIONS FOR MODEL EVALUATION

# Evaluate the model on ONE class
def get_result(
    model_constructor, 
    model_params: dict, 
    class_name: str, 
    base_path: str = "/content/"):

  # Paths preparation
  temp_path = os.path.join(base_path, class_name, class_name) # ex. /content/bottle/bottle
  train_path, test_path  = os.path.join(temp_path, "train", "good"), os.path.join(temp_path, "test")
  train_paths, test_paths= [train_path], [os.path.join(test_path, path) for path in os.listdir(test_path)]

  # Train & Evaluate
  model = model_constructor(**model_params)
  model.fit(train_paths)
  model.evaluate(test_paths, validation_flag = True)

  # Save model results
  result = {}
  result["cm"] = model.cm
  result["prfs"] = model.prfs
  result["auc_img"] = model.auc_img
  result["auc_pxl"] = model.auc_pxl

  return result

# Evaluate the model on ALL the MVTec classes
def get_results(
    model_constructor, 
    model_params: dict 
    ):

  results = {}
  misclassified = avg_img = avg_pxl = 0

  for class_name in mvtec_classes:
    print()
    print(f'Class: {class_name}')
    result = get_result(model_constructor, model_params, class_name)
    results[class_name] = result

    # Average computation
    misclassified = misclassified + result["cm"][0][1] + result["cm"][1][0]
    avg_img = avg_img + result['auc_img']
    avg_pxl = avg_pxl + result['auc_pxl']
  
  avg_auc_img = avg_img/len(results)
  avg_auc_pxl = avg_pxl/len(results)

  results["avg_auc_img"] = avg_auc_img
  results["avg_auc_pxl"] = avg_auc_pxl

  results["misclassified"] = misclassified
  results["model_params"] = model_params
  
  return results

# Print the results in a prettier way
def print_results(
    results: dict           # output of get_results
    ):
  
  print("\n\nCLASS BREAKDOWN")

  for className in mvtec_classes:
    result = results[className]
    print(f"ROCAUC img: {result['auc_img']:.3f}\tROCAUC pxl: {result['auc_pxl']:.3f}\tf1_score: {result['prfs'][2]:.3f} \t{className}")

  print("\nSUMMARY")
  print(f"Avg ROCAUC img: {results['avg_auc_img']:.3f}\nAvg ROCAUC pxl: {results['avg_auc_pxl']:.3f}\nTotal Misclassified: {results['misclassified']}")

# Saves the results object in a json file
def save_json(
    results: dict,          # output of get_results
    json_name: str          # name of the json file in output
    ) -> dict:
  
  results_json = {}

  for key in results:
    if key in mvtec_classes:
      results_json[key] = {
          "cm": results[key]["cm"].tolist(),
          "prfs": results[key]["prfs"],
          "auc_img": results[key]["auc_img"],
          "auc_pxl": results[key]["auc_pxl"]
      }
    else:
      match key:
        case "misclassified":
          results_json[key] = results[key].item()
        case "avg_auc_img":
          results_json[key] = results[key].item()
        case "avg_auc_pxl":
          results_json[key] = results[key].item()
        case _:
          results_json[key] = results[key]


  with open(json_name, 'w', encoding='utf-8') as f:
    json.dump(results_json, f, indent=4)

  return results_json

'''
def get_layer_results(model_constructor, model_params):
  layer_results = {}
  for layer in model_params["layers"]:
    print()
    print(f"Layer: {layer}")

    results = get_results(model_constructor, model_params)
    print_results(results)

    layer_results[layer] = results

  return layer_results
'''