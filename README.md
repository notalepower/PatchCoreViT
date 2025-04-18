# PatchCore Transformed 
A Vision Transformer Approach To Anomaly Detection
> ** NB: This repository is still under development **

### Models:
- PatchCore = Abstract Class
- VanillaPatchCore = PatchCore + CNN
- PatchCoreV2 = PatchCore + ViT
- PatchCoreV3 = PatchCore + ViT + Layer Concatenation
- PatchCoreSWin = PatchCore + SWin
- PatchCoreSWinV2 = PatchCore + SWin
---
## TODO
- [ ] Controllare in fit il Coreset Subsampling
- [ ] Implementare PatchCore + SWin
- [ ] Per ogni parametro in input delle funzioni aggiungere anche il tipo di input che si aspetta
- [ ] Aggiungere commenti in inglese nel codice
- [ ] Sistemare le cosine functions
- [ ] Completare l'anomalia a livello di pixel
---
## MvTecDataset Class
This custom class was tailored to fit the directory topography of the [MvTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)

> ### MvTec Directory Topography:
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
> - self.model: Backbone used to extract the features.
> - self.processor: 
> - self.features
> - self.memory_bank
> - self.f_coreset
> - self.eps_coreset
> - self.k_nearest
> - self.image_size

### Methods:
> #### __init__ (self, f_coreset, eps_coreset, k_nearest, vanilla, backbone, image_size)
>- Inizializzazione dei parametri iniziali, preparazione della gpu
>- hook salva l'output in self.features
>- register_forward_hook(hook) viene usato per associare ad un layer il suo hook

>#### forward (self, sample: tensor)
>- Fa passare l'input dentro self.model
>- Restituisce self.features, ovvero l'ouput dei layer ai quali la hook è stata agganciata

>#### extract_embeddings (self, sample)
>- All'interno utilizza la forward per ottenere le feature maps
>- Queste verranno lavorate e trasformate in base al self.model
>- Restituisce patch 
>- Restituisce feature_maps per questioni di debug

>#### fit (self, train_dataloader: DataLoader,  scale: int=1) -> None:
>- Training della rete, all'interno viene chiamata extract_embeddings
>- Patch vengono salvati all'interno del memmory bank
>- Viene applicato il coreset subsampling in cui solo i patch più significativi vengono mantenuti

>#### evaluate (self, test_dataloader: DataLoader, validation_flag = True):
>- Calcola F1 score

### Other:
- save_memory_bank: salva la memory bank per evitare di fare la train ogni volta.
- load_memory_bank: carica la memory bank 
- inference: funzione usata da Manuel