# PatchCore Transformed 
[A Vision Transformer Approach To Anomaly Detection](https://amslaurea.unibo.it/id/eprint/31435/)
> ** N.B.: This repository is still under development **

This project was part of my master's thesis. It focused on integrating ViTs into PatchCore as an alternative to traditional CNNs.


### Models:
- PatchCore = Abstract Class
- VanillaPatchCore = PatchCore + CNN Backbone
- PatchCoreViT = PatchCore + ViT Backbone ( + Layer Concatenation) 
- PatchCoreSWin = PatchCore + SWin Backbone ( + Layer Concatenation) 

### Patchcore
Visual representation of PatchCore's Algorithm

![PatchCore Algorithm](media/patch_analysis.gif)

---
## Classes
### MvTecDataset Class
This custom class was tailored to fit the directory topography of the [MvTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads).

> #### MvTec Directory Topography:
> - / object_name
>   - / train 
>       - / good / *.png
>   - / test
>       - / good / *.png
>       - / nok_1 / *.png
>       - ...
>       - / nok_N / *.png
>   - / ground_truth / 
>       - / nok_1 / *_mask.png
>       -  ...
>       - / nok_N / *_mask.png

> ### Attributes:
> - self.filepaths: List containing the absolute path of the images.
> - self.target: List containing the ok and not ok (nok).
> - self.transform: The Image transformation that needs to be applied. 
---
## PatchCore Class
>### Attributes:
> - self.device
> - self.model
> - self.processor:
> - self.features
> - self.memory_bank
> - self.f_coreset
> - self.eps_coreset
> - self.k_nearest
> - self.image_size

### Methods:
> #### __init__ (self, f_coreset, eps_coreset, k_nearest, vanilla, backbone, image_size)
>- Parameter initialization, it also prepares te GPU
>- hook saves the output in self.features
>- register_forward_hook(hook): used to associate a layer with a hook function

>#### forward (self, sample: tensor)
>- Passes the input through the backbone (self.model)
>- Returns self.features, which is the output from the layers to which the hook was attached.

>#### extract_embeddings (self, sample)
>- It uses forward method to obtain the feature maps
>- Feature maps will be processed and transformed based on the type of backbone (self.model)
>- Returns a patch
>- Returns feature_maps (for debugging purposes)

>#### fit (self, train_dataloader: DataLoader,  scale: int=1) -> None:
>- Network training, inside it extract_embeddings is called.
>- Patches are saved inside the memory bank (self.memory_bank)
>- Coreset subsampling prunes the memory bank by keeping the most significant patches.

>#### evaluate (self, test_dataloader: DataLoader, validation_flag = True):
>- Compute F1 score

### TODO:
- [ ] Fix function Inference (implemented by Manuel)
- [ ] For each input parameter add the types
- [ ] Add English comments in the code
- [ ] Fix cosine functions
- [ ] Complete anomaly at pixel level