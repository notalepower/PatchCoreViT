import os
from xmlrpc.client import boolean
import cv2
import PIL 
import math
import shutil
import time
import numpy as np
import torch
from typing import List, Tuple
from PIL import Image 
from PIL import ImageFilter
from torch import tensor
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from tqdm import tqdm
from abc import ABC, abstractmethod

from sklearn import random_projection
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import f1_score, roc_curve, roc_auc_score

from transformers import AutoModel, AutoImageProcessor, BaseImageProcessor

### UTILITY ###

# MVTecDataset DONE
class MVTecDataset(torch.utils.data.Dataset):  

    def __init__(self, folders_path: List[str], processor: BaseImageProcessor, resize: int = 256, cropsize : int = 224):

        # List subfolders
        self.filepaths = []
        self.target = [] # 0 Good, 1 Nok
        self.transform = processor
        self.cropsize = cropsize
        self.resize = resize

        # Populate the target and filepath lists
        for folder in folders_path:         # cycle the list of folders
            for name in os.listdir(folder): # cycle the photos in the folder
                filename = os.path.join(folder, name)
                
                self.filepaths.append(filename)
                self.target.append(0 if "good" in filename else 1)
        self.target = torch.tensor(self.target, dtype=torch.int32) # transform target into tensor

        self.transform_mask = T.Compose([
            T.Resize(self.resize, Image.NEAREST),   # resizes image if it is too big
            T.CenterCrop(self.cropsize),            # apply center crop
            T.ToTensor(),                           # converts it to a tensor
        ])

    def __getitem__(self, index: int) -> Tuple[tensor, int, str, tensor]: 
        filepath = self.filepaths[index]
        x = self.transform(Image.open(filepath).convert("RGB"))
        y = self.target[index]

        if y == 0:  # creates a mask of zeros since there are no anomalies
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:       # otherwise it retrieves it
            filepath_mask = filepath.replace("test", "ground_truth").replace(".","_mask.")
            mask = self.transform_mask(Image.open(filepath_mask)) if os.path.exists(filepath_mask) else torch.zeros([1, self.cropsize, self.cropsize])

        return x, y, filepath, mask

    def __len__(self) -> int:
        return len(self.filepaths)

# Guassian Blur DONE
def gaussian_blur(img: tensor, radius: int = 4) -> tensor:
    """
        Apply a gaussian smoothing with sigma = 4 over the input image.
    """
    # Setup
    blur_kernel = ImageFilter.GaussianBlur(radius=radius)
    tensor_to_pil, pil_to_tensor = T.ToPILImage(), T.ToTensor()

    # Smoothing
    max_value = img.max() # Maximum value of all elements in the image tensor
    blurred_pil = tensor_to_pil(img[0] / max_value).filter(blur_kernel)
    blurred_map = pil_to_tensor(blurred_pil) * max_value

    return blurred_map

# Get best threshold DONE
def get_y_f1score(y_true: List[int], y_hat: List[float], threshold: float) -> Tuple[List[float], float]:
    y_pred_hat = [(1 if score > threshold else 0) for score in y_hat]
    f1score = f1_score(y_true, y_pred_hat)
    return y_pred_hat, f1score

def get_best_threshold(y_true: List[int], y_hat: List[float], initial_threshold: float, search: int) -> Tuple[List[int], float, float, float]: 
    increment = 0.01
    thresholds = np.arange(initial_threshold-search, initial_threshold+search + increment, increment)
    
    intial_y_hat, initial_f1score = get_y_f1score(y_true, y_hat, initial_threshold)
    
    optimal_threshold = initial_threshold
    optimal_f1score = initial_f1score
    optimal_y_hat = intial_y_hat

    for threshold in thresholds:
        y_pred_hat, f1score = get_y_f1score(y_true, y_hat, threshold)
        if f1score > optimal_f1score:
            optimal_y_hat = y_pred_hat
            optimal_f1score = f1score
            optimal_threshold = threshold
    
    return optimal_y_hat, optimal_f1score, optimal_threshold, initial_f1score

################

# PatchCore
backbones = [
    "timm/wide_resnet50_2.tv2_in1k",
    "google/vit-base-patch16-224-in21k"
]

class PatchCore(torch.nn.Module, ABC): # Abstract class
    
    def hook(self, module, input, output) -> None: # Hook to extract feature maps
        """This hook saves the extracted feature map on self.featured."""
        self.features.append(output)
        
    @abstractmethod
    def set_hooks(self):
        """
            This method must be implemented by the subclass
            es:
                self.model.encoder.layer[self.l1].layernorm_after.register_forward_hook(hook)   # registration of the hook
                self.model.encoder.layer[self.l2].layernorm_after.register_forward_hook(hook)   # registration of the hook
        """
        pass

    def __init__(
            self,
            layers: List[int] = [1,2],
            backbone: str = 'google/vit-base-patch16-224-in21k',
            f_coreset: float = 1,       # Fraction rate of training samples
            eps_coreset: float = 0.09,  # SparseProjector parameter
            k_nearest: int = 9,         # k parameter for K-NN search
            seed: int = 42
        ):

        assert f_coreset > 0
        assert eps_coreset > 0
        assert k_nearest > 0

        super().__init__()
        torch.manual_seed(seed)

        # Model creation
        self.model = AutoModel.from_pretrained(backbone)
        self.processor = AutoImageProcessor.from_pretrained(backbone)
        self.layers = layers
        self.set_hooks()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.seed = seed
        print(f"[INFO][__init__] Model PatchCore loaded on device: {self.device}")

        # Disable gradient computation
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Parameters
        self.memory_bank = []
        self.f_coreset = f_coreset      # Fraction rate of training samples
        self.eps_coreset = eps_coreset  # SparseProjector parameter
        self.k_nearest = k_nearest      # k parameter for K-NN search
        self.threshold = None

    def get_sample(self, input_path:str):
        img = Image.open(input_path).convert("RGB")
        sample = self.processor(img)
        sample.pixel_values = [torch.Tensor(sample['pixel_values'][0]).unsqueeze(0)]
        return sample

    @abstractmethod
    def forward(self, sample: tensor) -> tensor:
        """
            This method must be implemented by the subclass
            es:
                self.features = []
                _ = self.model(sample, output_attentions=True, output_hidden_states=True, return_dict=True)
                return self.features
        """
        pass

    @abstractmethod
    def extract_embeddings(self, sample: tensor) -> Tuple[tensor, tensor]:
        """
            This method must be implemented by the subclass
            * insert and example *
        """
        pass

    # Get coreset: TO_STUDY 
    def get_coreset(self, memory_bank: tensor, l: int = 1000, eps: float = 0.90) -> tensor:

        coreset_idx = []  # Returned coreset indexes
        idx = 0

        # Fitting random projections
        try:
            seed = self.seed
            transformer = random_projection.SparseRandomProjection(eps=eps, random_state=seed)
            memory_bank = torch.tensor(transformer.fit_transform(memory_bank))
        except ValueError:
            print("Error: could not project vectors. Please increase `eps`.")

        # Coreset subsampling
        print(f'Start Coreset Subsampling...')

        last_item = memory_bank[idx: idx + 1]   # First patch selected = patch on top of memory bank
        coreset_idx.append(torch.tensor(idx))
        min_distances = torch.linalg.norm(memory_bank - last_item, dim=1, keepdims=True)    # Norm l2 of distances (tensor)

        # If possible move to GPU the items
        if torch.cuda.is_available():
            last_item = last_item.to(self.device)
            memory_bank = memory_bank.to(self.device)
            min_distances = min_distances.to(self.device)

        for _ in tqdm(range(l - 1)):
            distances = torch.linalg.norm(memory_bank - last_item, dim=1, keepdims=True)    # L2 norm of distances (tensor)
            min_distances = torch.minimum(distances, min_distances)                         # Verical tensor of minimum norms
            idx = torch.argmax(min_distances)                                               # Index of maximum related to the minimum of norms

            last_item = memory_bank[idx: idx + 1]   # last_item = maximum patch just found
            min_distances[idx] = 0                  # Zeroing last_item distances
            coreset_idx.append(idx.to("cpu"))       # Save idx inside the coreset

        return torch.stack(coreset_idx)

    def fit(self, train_paths: List[str], scale: int = 1) -> None:
        
        self.memory_bank_paths = []

        train_dataloader = self.get_dataloader(train_paths)
        tot = len(train_dataloader) // scale
        counter = 0
        for sample, _, file_path, _ in tqdm(train_dataloader, total=tot):
      
            patch, _ = self.extract_embeddings(sample)
            self.memory_bank.append(patch.cpu())                # Fill memory bank
            self.memory_bank_paths.append(file_path)
            counter += 1
            if counter > tot:
                break

        self.memory_bank = torch.cat(self.memory_bank, 0) # VStack the patches        

        # TODO: Study Coreset subsampling
        if self.f_coreset < 1:
            self.memory_bank = self.memory_bank.detach()                # Removes from GPU
            coreset_idx = self.get_coreset(
                self.memory_bank,
                l = int(self.f_coreset * self.memory_bank.shape[0]),
                eps = self.eps_coreset
            )

            self.memory_bank = self.memory_bank[coreset_idx]            # Filters the relevant patches
        
        self.memory_bank = self.memory_bank.to(self.device)             # Restores to GPU

    # Cross Euclidean distance
    def cdist(self, patch, memory_bank):
        return torch.cdist(patch, memory_bank, p=2.0)

    # Cross Cosine similarity distance
    def sdist(self, a:tensor, b:tensor) -> tensor:
        # a, b must be (1,1536) and (10, 1536)
        # a = model.patch
        # b = model.memory_bank.view(1960,-1)

        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        res = torch.mm(a_norm, b_norm.transpose(0,1))
        return 1 - res

    def predict(self, sample: tensor, compute_distance = cdist)->Tuple[tensor, tensor]:
        
        # Patch extraction
        patch, _ = self.extract_embeddings(sample)
        n_patches, hidden_size = patch.shape 

        # Compute maximum distance score s* (equation 6 from the paper)
        distances = compute_distance(patch, self.memory_bank)         # L2 norm dist btw test patch with each patch of memory bank
        dist_score, dist_score_idxs = torch.min(distances, dim=1)       # Val and index of the distance scores (minimum values of each row in distances)
        s_idx = torch.argmax(dist_score)                                # Index of the anomaly candidate patch
        s_star = torch.max(dist_score)                                  # Maximum distance score s*
        m_test_star = torch.unsqueeze(patch[s_idx], dim=0)              # Anomaly candidate patch
        m_star = self.memory_bank[dist_score_idxs[s_idx]].unsqueeze(0)  # Memory bank patch closest neighbour to m_test_star

        # KNN
        knn_dists = torch.cdist(m_star, self.memory_bank, p=2.0)        # L2 norm dist btw m_star with each patch of memory bank
        _, nn_idxs = knn_dists.topk(k=self.k_nearest, largest=False)    # Values and indexes of the k smallest elements of knn_dists

        # Computes the weight w
        m_star_neighbourhood = self.memory_bank[nn_idxs[0, 1:]]         
        w_denominator = torch.linalg.norm(m_test_star - m_star_neighbourhood, dim=1)    # Sum over the exp of l2 norm distances btw m_test_star and the m* neighbourhood
        norm = torch.sqrt(torch.tensor(hidden_size))                                    # Softmax normalization trick to prevent exp(norm) from becoming infinite
        
        w = 1 - (torch.exp(s_star / norm) / torch.sum(torch.exp(w_denominator / norm))) # Equation 7 from the paper
        
        # Compute image-level anomaly score s
        s = w * s_star

        # # Segmentation map
        height = width = int(math.sqrt(n_patches))
        fmap_size = (height, width)                     # Feature map sizes: h, w
        segm_map = dist_score.view(1, 1, *fmap_size)    # Reshape distance scores tensor
        segm_map = torch.nn.functional.interpolate(     # Upscale by bi-linaer interpolation to match the original input resolution
                        segm_map,
                        size=(self.image_size, self.image_size),
                        mode='bilinear'
                    )
        segm_map = gaussian_blur(segm_map.cpu())         # Gaussian blur of kernel width = 4
        
        # debugging purposes
        self.s_idx = s_idx
        self.distances = distances
        self.dist_score = dist_score
        self.dist_score_idxs = dist_score_idxs
        self.s_star = s_star
        self.w = w
        self.target_idx = dist_score_idxs[s_idx]
        self.patch = patch

        self.knn_dists = knn_dists
        self.nn_idxs = nn_idxs
        
        self.score = s
        self.segm_map= segm_map

        return s, segm_map

    def compute_ROC_AUC_score(self, ground_truth, predictions, validation_flag: boolean, title: str="IMAGE"):
        if len(np.unique(predictions)) == 1:
            print(f"roc_auc_score at {title} level can't be computed, {title}_labels (y_true) doesn't contain anomalies")
            return -1
        else:
            result = roc_auc_score(ground_truth, predictions)
            print(f"{'Val' if validation_flag else 'Test'}: {title} Level ROCAUC: {result:.3f}")
            return result
        
    def evaluate(self, test_paths: List[str], compute_distance = cdist, validation_flag: boolean = True):
  
        image_preds, image_labels = [], []
        pixel_preds, pixel_labels = [], []
        time_list = []
                
        test_dataloader = self.get_dataloader(test_paths)
        for sample, label, _, mask in tqdm(test_dataloader):
            
            start_time = time.time()
            score, segm_map = self.predict(sample, compute_distance)  # Anomaly Detection
            end_time = time.time()

            elapsed_time = end_time - start_time
            time_list.append(elapsed_time)
            # print(f"Elapsed time: {elapsed_time:.3f} seconds")

            image_labels.append(label)
            image_preds.append(score.detach().cpu().numpy())
            pixel_labels.extend(mask.flatten().numpy())
            pixel_preds.extend(segm_map.flatten().cpu().numpy())

        image_labels = np.stack(image_labels)
        image_preds = np.stack(image_preds)
        
        # Check ROC AUC at IMAGE level
        image_level_rocauc = self.compute_ROC_AUC_score(image_labels, image_preds, validation_flag, "IMAGE")

        # Check ROC AUC at PIXEL level
        pixel_level_rocauc = self.compute_ROC_AUC_score(pixel_labels, pixel_preds, validation_flag, "PIXEL")
        
        # calculate image-level ROC AUC score
        # set initial threshold
        if len(np.unique(image_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(image_labels, image_preds)
            distances = np.sqrt(( 1 - tpr )**2 + fpr**2 )        # Euclidean distance between points and (0, 1) which is the perfect score
            best_index = np.argmin(distances)
            initial_score_threshold = thresholds[best_index]
            print('[INFO][evaluate] Image Level ROCAUC: %.3f' % (image_level_rocauc))
            search=10 # -1.5 initial_threshold +1.5
        else:
            initial_score_threshold = 0
            search=10 # -10 0 +10
        
        ## Find best threshold value
        if validation_flag:
            optimal_y_hat, optimal_score_f1score, optimal_score_threshold, initial_score_f1score = get_best_threshold(image_labels, image_preds, initial_score_threshold, search)
            print(f"[INFO][evaluate] Initial Score Threshold: {initial_score_threshold:.3f} F1Score: {initial_score_f1score:.3f}")
            print(f"[INFO][evaluate] Optimal Score Threshold: {optimal_score_threshold:.3f} F1Score: {optimal_score_f1score:.3f}")
            print(f"[INFO][evaluate] Average Inference time with batch_size={test_dataloader.batch_size}: {np.mean(time_list):.3f}s")
            self.threshold = optimal_score_threshold
        
            # For debugging purposes
            self.norm_predictions = optimal_y_hat
            self.cm = confusion_matrix(image_labels, self.norm_predictions)
            self.prfs = precision_recall_fscore_support(image_labels, self.norm_predictions, average = 'binary')
        
        # For debugging purposes
        self.ground_truths = image_labels
        self.predictions = image_preds
        self.segm_maps = pixel_preds
        
        self.auc_img = image_level_rocauc
        self.auc_pxl = pixel_level_rocauc

    # Utilies Functions
    def get_dataloader(self, paths: List[str]) -> DataLoader:
        dataset = MVTecDataset(paths, self.processor, self.image_size)
        return DataLoader(dataset)

    def save_memory_bank(self, path: str): # Save memory bank
        torch.save(self.memory_bank, path)

    def load_memory_bank(self, path: str): # Load memory bank
        self.memory_bank = torch.load(path)
        self.memory_bank = self.memory_bank.to(self.device) # Load to GPU

class VanillaPatchCore(PatchCore): # CNN backbone with layer concatenation
    
    # Override
    def set_hooks(self):
        for layer in self.layers:
            match layer:
                case "layer1":
                    self.model.timm_model.layer1[-1].register_forward_hook(self.hook)          
                case "layer2":
                    self.model.timm_model.layer2[-1].register_forward_hook(self.hook)          
                case "layer3":
                    self.model.timm_model.layer3[-1].register_forward_hook(self.hook)          
                case "layer4":
                    self.model.timm_model.layer4[-1].register_forward_hook(self.hook)          

    def __init__(
            self,
            layers = None,
            backbone: str = 'timm/wide_resnet50_2.tv2_in1k',
            f_coreset:float = 1,    
            eps_coreset: float = 0.90, 
            k_nearest: int = 9,        
            seed: int = 42
    ):
        assert f_coreset > 0
        assert eps_coreset > 0
        assert k_nearest > 0

        super().__init__(layers, backbone, f_coreset, eps_coreset, k_nearest, seed)
        self.image_size = self.processor.data_config['input_size'][-1]
        self.avg = torch.nn.AvgPool2d(3, stride=1) # For shrinking the spatial dimensions        

    # Override
    def forward(self, sample: tensor):
        self.features = []
        _ = self.model(sample)
        return self.features

    # Override
    def extract_embeddings(self, sample: tensor)-> Tuple[tensor, tensor]:
            sample_preprocessed = sample.pixel_values[0]
            feature_maps = self(sample_preprocessed.to(self.device))      # Extract feature maps
            fmap_size = feature_maps[0].shape[-2]         # fmap_size = 28
            self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)  ## For stretching the spatial dimensions
            
            resized_maps = [self.resize(self.avg(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)            # Merge the resized feature maps
            patch = patch.reshape(patch.shape[1], -1).T   # Craete a column tensor
            return patch, feature_maps
    
    # TODO: Questo lo puoi controllare per ultimissimo
    def inference(self, folders_input, folder_results,  score_threshold, scoremap_threshold, transform = None):
        
        # TODO USE THE DEFAULT ONE
        if transform is None:
            transform = T.Compose([
                T.ToPILImage(),
                # T.GaussianBlur(5, sigma=2),
                # T.ColorJitter(brightness=(0.5, 0.5)),
                # T.Grayscale(num_output_channels=3),
                # T.RandomPosterize(4, p=1),
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
          
        score_map_list = []
        score_list = []
        y_pred_list = []
        y_true_list = []
        fp_list = []
        fn_list = []
        tp_list = []
        tn_list = []
        time_list = []
        for folder_in in folders_input:
            folder_name = folder_in.split("/")[-1]
            folder_out = f"{folder_results}/{folder_name}"
            if os.path.exists(folder_out):
                shutil.rmtree(folder_out)	
            os.makedirs(folder_out)
            print(folder_in)
            filenames = [os.path.join(folder_in, name) for name in os.listdir(folder_in)]
            for filename in tqdm(filenames):
                name = os.path.basename(filename)
                #print (filename)
                im = Image.open(filename).convert("RGB")
                im_np = np.array(im)
                #####
                # center = (im_np.shape[0]//2, im_np.shape[1]//2) 
                # im_np = cv2.circle(im_np, center, 30, (255,255,255), -1)
                #####
                im_t = transform(im_np)    
                sample = im_t.unsqueeze(0)
                
                # TODO GPU
                sample = sample.to(self.device)            
                score, score_map = self.predict(sample)  # Anomaly Detection

                score = score.cpu().numpy()
                score_map = score_map.cpu().numpy()[0,:,:]  # 1x224x224 # TODO if process batch  modify
                
                score_list.append(score)
                score_map_list.extend(score_map.flatten())
                ###
                self.im_np = im_np
                self.score_map = score_map
                ###
                out = get_results(im_np, score_map, threshold=scoremap_threshold)  
                
                # Chech score
                y_pred = 0
                res = "Good"
                if score > score_threshold:
                    y_pred = 1
                    res = "Anomaly"

                # check ideal
                y_true = 1
                if "good" in str(filename):
                    y_true = 0
                    
                y_pred_list.append(y_pred)
                y_true_list.append(y_true)
                
                err = False
                if y_true==0 and y_pred==1:
                    #  False Negative: Ho predetto nok= y_pred=1 ma in realtà era good y_true=0
                    fn_list.append(str(filename))
                    err = True
                elif y_true==1 and y_pred==0:
                    #  False Positive: Ho predetto good= y_pred=0 ma in realtà era nok y_true=1
                    fp_list.append(str(filename))
                    err = True
                elif y_true==0 and y_pred==0:
                    #  True Negative: Ho predetto good= y_pred=0 e in realtà è good y_true=0
                    tn_list.append(str(filename))
                else:
                    #  True Positive: Ho predetto npk= y_pred=1 e in realtà è nok y_true=1
                    tp_list.append(str(filename))
                                    

                folder = filename.split("/")[-2]
                f = os.path.basename(filename)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(out, f'Score: {score:.3f} Result: {res}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                filepath_out = os.path.join(folder_out,f)
                #print(filepath_out)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath_out, out)


        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        TP = len(tp_list)
        TN = len(tn_list)
        FN = len(fn_list)
        FP = len(fp_list)
        recall = TP/(TP+FN) #recall_score(y_true, y_pred)
        precision = TP/(TP+FP) #precision_score(y_true, y_pred)
        f1score = 2 * (precision * recall) / (precision + recall) # f1_score(y_true, y_pred)
        fp = '\n'.join(fp_list)
        fn = '\n'.join(fn_list)
        code_to_write=f"""

        --------------------------------------------------------------------------------------------


        --------------------------------------------------------------------------------------------
        Precision:  TP/(TP+FP) = {precision*100:.2f} %
        Percentuale che mi dice quanto l'algoritmo e bravo a individuare i casi anomali
        FP: e un  numero che indica quanto volte l'algoritmo ha predetto good ma in realte era nok

        Recall TP/(TP+FN) = {recall*100:.2f} %
        Percentuale che mi dice quanto l'algoritmo e bravo a individuare i casi non anomali
        FN: e un  numero che indica quanto volte l'algoritmo ha predetto nok ma in realte era good

        F1 Score = 2* (Precision*Recall) / (Precision+Recall) = {f1score*100:.2f} %
        Metrica che bilancia precision e recall
        --------------------------------------------------------------------------------------------

        --------------------------------------------------------------------------------------------
                    Actual/Predicted    | Predicted Anomaly (1)  | Predicted Good (0)"
                    Actual anomaly (1)  |        TP={TP}         |       FP={FP} 
                    Actual good    (0)  |        FN={FN}         |       TN={TN}    
        --------------------------------------------------------------------------------------------

        --------------------------------------------------------------------------------------------
        [INFO][evaluate][False Positive] Ho predetto good= y_pred=0 ma in realte era nok y_true=1
        {fp}
        --------------------------------------------------------------------------------------------

        --------------------------------------------------------------------------------------------
        [INFO][evaluate][False Negative] Ho predetto nok= y_pred=1 ma in realtà era good y_true=0
        {fn}
        --------------------------------------------------------------------------------------------
        """
        filepath_results=f"{folder_results}/results.txt"
        with open(filepath_results, 'w') as file:
            file.write(code_to_write)


        with open(filepath_results, 'r') as f:
            # Read the contents of the file and print them line by line
            for line in f:
                print(line, end='')  # end='' avo

class PatchCoreViT(PatchCore): # ViT backbone with layer concatenation
    
    # Override
    def set_hooks(self):
        [ self.model.encoder.layer[layer].layernorm_after.register_forward_hook(self.hook) for layer in self.layers ]
             
    def __init__(
            self,
            layers: List[int],
            backbone: str = "google/vit-base-patch16-224-in21k",
            f_coreset: float = 1,    
            eps_coreset: float = 0.90, 
            k_nearest: int = 9,        
            seed: int = 42
    ):

        super().__init__(layers, backbone, f_coreset, eps_coreset, k_nearest, seed)
        self.image_size = self.processor.size["height"]

    # Override
    def forward(self, sample: tensor):
        self.features = []
        _ = self.model(sample, output_hidden_states=True, return_dict=True)
        return self.features
    
    # Override   
    def extract_embeddings(self, sample):
        sample_preprocessed = sample.pixel_values[0]
        output = self(sample_preprocessed.to(self.device))
        output = [ torch.split(o, [1, o.shape[1] - 1], dim=1)[1] for o in output ]  # removes classification token in front of the maps
        feature_maps = torch.cat(output, 2) # concatenates the feature's levels
        patch = feature_maps.squeeze()
        return patch, feature_maps
    
    # TODO: Questo lo puoi controllare per ultimissimo
    def inference(self, folders_input, folder_results,  score_threshold, scoremap_threshold, transform = None):
        
        # TODO: USE THE DEFAULT ONE
        if transform is None:
            transform = T.Compose([
                T.ToPILImage(),
                # T.GaussianBlur(5, sigma=2),
                # T.ColorJitter(brightness=(0.5, 0.5)),
                # T.Grayscale(num_output_channels=3),
                # T.RandomPosterize(4, p=1),
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
          
        score_map_list = []
        score_list = []
        y_pred_list = []
        y_true_list = []
        fp_list = []
        fn_list = []
        tp_list = []
        tn_list = []
        time_list = []
        for folder_in in folders_input:
            folder_name = folder_in.split("/")[-1]
            folder_out = f"{folder_results}/{folder_name}"
            if os.path.exists(folder_out):
                shutil.rmtree(folder_out)	
            os.makedirs(folder_out)
            print(folder_in)
            filenames = [os.path.join(folder_in, name) for name in os.listdir(folder_in)]
            for filename in tqdm(filenames):
                name = os.path.basename(filename)
                #print (filename)
                im = Image.open(filename).convert("RGB")
                im_np = np.array(im)
                sample_preprocessed = torch.Tensor(transform(im).pixel_values[0]).to(self.device).unsqueeze(0)
                score, score_map = self.predict(sample_preprocessed)  # Anomaly Detection

                score = score.cpu().numpy()
                score_map = score_map.cpu().numpy()[0,:,:]  # 1x224x224 # TODO if process batch  modify
                
                score_list.append(score)
                score_map_list.extend(score_map.flatten())

                out = get_results(im_np, score_map, threshold=scoremap_threshold)  
                
                # Chech score
                y_pred = 0
                res = "Good"
                if score > score_threshold:
                    y_pred = 1
                    res = "Anomaly"

                # check ideal
                y_true = 1
                if "good" in str(filename):
                    y_true = 0
                    
                y_pred_list.append(y_pred)
                y_true_list.append(y_true)
                
                err = False
                if y_true==0 and y_pred==1:
                    #  False Negative: Ho predetto nok= y_pred=1 ma in realtà era good y_true=0
                    fn_list.append(str(filename))
                    err = True
                elif y_true==1 and y_pred==0:
                    #  False Positive: Ho predetto good= y_pred=0 ma in realtà era nok y_true=1
                    fp_list.append(str(filename))
                    err = True
                elif y_true==0 and y_pred==0:
                    #  True Negative: Ho predetto good= y_pred=0 e in realtà è good y_true=0
                    tn_list.append(str(filename))
                else:
                    #  True Positive: Ho predetto npk= y_pred=1 e in realtà è nok y_true=1
                    tp_list.append(str(filename))
                                    

                folder = filename.split("/")[-2]
                f = os.path.basename(filename)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(out, f'Score: {score:.3f} Result: {res}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                filepath_out = os.path.join(folder_out,f)
                #print(filepath_out)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath_out, out)

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        TP = len(tp_list)
        TN = len(tn_list)
        FN = len(fn_list)
        FP = len(fp_list)
        recall = TP/(TP+FN) #recall_score(y_true, y_pred)
        precision = TP/(TP+FP) #precision_score(y_true, y_pred)
        f1score = 2 * (precision * recall) / (precision + recall) # f1_score(y_true, y_pred)
        fp = '\n'.join(fp_list)
        fn = '\n'.join(fn_list)
        code_to_write=f"""

        --------------------------------------------------------------------------------------------


        --------------------------------------------------------------------------------------------
        Precision:  TP/(TP+FP) = {precision*100:.2f} %
        Percentuale che mi dice quanto l'algoritmo è bravo a individuare i casi anomali
        FP: è un  numero che indica quanto volte l'algoritmo ha predetto good ma in realtà era nok

        Recall TP/(TP+FN) = {recall*100:.2f} %
        Percentuale che mi dice quanto l'algoritmo è bravo a individuare i casi non anomali
        FN: è un  numero che indica quanto volte l'algoritmo ha predetto nok ma in realtà era good

        F1 Score = 2* (Precision*Recall) / (Precision+Recall) = {f1score*100:.2f} %
        Metrica che bilancia precision e recall
        --------------------------------------------------------------------------------------------

        --------------------------------------------------------------------------------------------
                    Actual/Predicted    | Predicted Anomaly (1)  | Predicted Good (0)"
                    Actual anomaly (1)  |        TP={TP}         |       FP={FP} 
                    Actual good    (0)  |        FN={FN}         |       TN={TN}    
        --------------------------------------------------------------------------------------------

        --------------------------------------------------------------------------------------------
        [INFO][evaluate][False Positive] Ho predetto good= y_pred=0 ma in realtà era nok y_true=1
        {fp}
        --------------------------------------------------------------------------------------------

        --------------------------------------------------------------------------------------------
        [INFO][evaluate][False Negative] Ho predetto nok= y_pred=1 ma in realtà era good y_true=0
        {fn}
        --------------------------------------------------------------------------------------------
        """
        filepath_results=f"{folder_results}/results.txt"
        with open(filepath_results, 'w') as file:
            file.write(code_to_write)


        with open(filepath_results, 'r') as f:
            # Read the contents of the file and print them line by line
            for line in f:
                print(line, end='')  # end='' avo

class PatchCoreSWin(PatchCore): # SWin backbone with layer concatenation
    
    # Override
    def set_hooks(self):
        for layer in self.layers:
            for block in self.blocks:
                self.model.encoder.layers[layer].blocks[block].register_forward_hook(self.hook)

    def __init__(
            self,
            layers: List[int] = [2],
            blocks: List[int] = [0, 1, 2, 3],
            backbone: str = "microsoft/swin-small-patch4-window7-224",
            f_coreset:float = 1, 
            eps_coreset: float = 0.90, 
            k_nearest: int = 9,        
            seed: int = 42
    ):

        self.blocks = blocks
        super().__init__(layers, backbone, f_coreset, eps_coreset, k_nearest, seed)
        self.image_size = self.processor.size['height']

    # Override
    def forward(self, sample: tensor):
        self.features = []
        _ = self.model(sample, return_dict=True)
        return self.features

    # Override          
    def extract_embeddings(self, sample):
        sample_preprocessed = sample.pixel_values[0]
        output = self(sample_preprocessed.to(self.device))
        features_maps = torch.cat([ o[0] for o in output ], dim = 2)
        # features_maps = torch.cat([features_maps], 2) # concatenates the feature's levels
        patch = features_maps.squeeze()
        return patch, features_maps
        

