# PatchCore Transformed 
[A Vision Transformer Approach To Anomaly Detection](https://amslaurea.unibo.it/id/eprint/31435/)
> ** N.B.: This repository is still under development **

This project was part of my master's thesis. It focused on integrating ViTs into PatchCore as an alternative to traditional CNNs.


### Models:
- PatchCore = Abstract Class
- VanillaPatchCore = PatchCore + CNN Backbone
- PatchCoreViT = PatchCore + ViT Backbone ( + Layer Concatenation) 
- PatchCoreSWin = PatchCore + SWin Backbone ( + Layer Concatenation) 

# Patchcore
## Visual representation of PatchCore's Algorithm

![PatchCore Algorithm](media/vit_patch_analysis.gif)

*The f_coreset parameter was set to 1 to prevent discarding any patches from the memory bank, esuring that the original image associated with the target patch could still be retrieved.*

*With a ViT backbone, an interesting property can be observed: the target patch often matches patches from the same spatial region in other images, thanks to the positional encodings.*

*For instance, a white patch in the bottom-right corner of an image is more likely to match a white patch in the bottom-right corner of another image in the memory bank, rather than a white patch in a different location such as the bottom-left.*

## Heatmap comparison

![Distance comparison](media/heatmap_comparison.png)

*The choice of distance affects the appeareance of PatchCore heatmaps.
Cosine similarity is bounded between 0 and 1, producing a well-defined heatmap with clear constrast, making anomalous regions easier to identify.
Euclidean distance, in contrast, is unbounded, resulting in a more spread-out range of values and a less distinct heatmap, where anomalies are visually less prominent.*

# patchcore_utils.py
## MvTecDataset Class
This custom class was tailored to fit the directory topography of the [MvTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads).

> ### MvTec Directory Topography:
> (*Ex. with 'bottle' class*)
> - / bottle
>   - / train 
>       - / good / *.png
>   - / test
>       - / good / *.png
>       - / broken_large / *.png
>       - ...
>       - / contamination / *.png
>   - / ground_truth / 
>       - / broken_large / *_mask.png
>       -  ...
>       - / contamination / *_mask.png

> ### Attributes:
> - self.filepaths: List containing the absolute path of the images.
> - self.target: List containing the ok and not ok (nok).
> - self.transform: The Image transformation that needs to be applied. 
---
## PatchCore Class
>### Attributes:
> - self.device
> - self.backbone
> - self.model
> - self.processor
> - self.layers
> - self.seed
> - self.memory_bank
> - self.f_coreset
> - self.eps_coreset
> - self.k_nearest
> - self.image_size

### Methods:
> #### __init__ (self, f_coreset, eps_coreset, k_nearest, vanilla, backbone, image_size):
>- Parameter initialization, it also prepares te GPU
>- hook saves the output in *self.features*.
>- register_forward_hook(hook): used to associate a layer with a hook function

>#### forward (self, sample: tensor):
>- Passes the input through the backbone (self.model).
>- Returns self.features, which is the output from the layers to which the hook was attached.

>#### extract_embeddings (self, sample):
>- It uses *forward* method to obtain the feature maps
>- Feature maps will be processed and transformed based on the type of backbone (self.model)
>- Returns a patch.
>- Returns feature_maps (for debugging purposes).

>#### predict (self, sample, metric):
>- Returns the anomaly score and the anomaly map. 
>- Uses the method *extract_emdeddings*

>#### fit (self, train_paths,  scale: int=1):
>- Populates the memory bank (*self.memory_bank*)
>- Network training, inside it *extract_embeddings* is called.
>- Coreset subsampling prunes the memory bank by keeping the most significant patches.

>#### evaluate (self, test_val_paths, metric, validation_flag = True):
>- Computes the ROCAUC at image and pixel levels.
>- Computes the precision, recall and F1 score.
>- The threshold for the F1 score was determined via *dynamic thresholding*. This approach involves testing different threshold values and selecting the one that produces the highest F1-score on the validation set.

---
# patchcore_utils.py
TODO: brief description on what it does

>### Attributes:
> - mvtec_classes
> - img_size
> - red_color
> - thickness
> - n_patch_img
> - n_patch_side
> - w_patch
> - h_patch

### Methods:
> #### plot_gridded_image(img_path):
>- Example item

> #### get_box_coordinates(idx):
>- Example item

> #### get_plot_images(idx, path):
>- Example item

> #### show(input_idx, input_path, model, distance_label, save, alpha):
>- Example item

> #### create_gif(input_path, model, metric, duration, output_path):
>- Example item

> #### get_result(model_constructor, model_params, class_name, base_path):
>- Example item

> #### get_results(model_constructor, model_params):
>- Example item

> #### print_results(results):
>- Example item

> #### save_json(results, json_name):
>- Example item


# TODO:
- [ ] For each input parameter add the types
- [ ] Add English comments in the code