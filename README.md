## Zero-Shot Performance Prediction for Probabilistic Scaling Laws  

:page_with_curl: **Paper & Resources**  
- :paperclip: [ArXiv Paper](https://www.arxiv.org/pdf/2510.16743)  
- :pushpin: [NeurIPS 2025 Paper Homepage](https://neurips.cc/virtual/2025/loc/san-diego/poster/115947)  
- :chart_with_upwards_trend: [NeurIPS 2025 Poster PDF](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/115947.png?t=1761032474.1345522)  
- :movie_camera: [NeurIPS 2025 Teaser](https://recorder-v3.slideslive.com/?share=103761&s=c820cd35-daed-45ca-a722-0758f3cb648a)  

---  

## Repository Overview  

This repository contains the code for our NeurIPS 2025 paper:  
**"Zero-Shot Performance Prediction for Probabilistic Scaling Laws"**  

:file_folder: **Folder containing all files for the main code:** `HMOGPLV`  

:scroll: **Find the requirements in**: `requirments.txt`.

---  

## :open_file_folder: Datasets  
Please find all datasets we used for our experiments in:

- **Scaling Law Experiments:**  
  - `NanoGPTlarge`: `Datasets/Dataset/init_data`  
  - `NanoGPTsmall`: `Datasets/Dataset/init_data2`  

- **Bilingual Experiments:**  
  - `Datasets/Dataset/bilingual`  

- **Multilingual Experiments:**  
  - `Datasets/Dataset/multilingual`  

---  

## :clipboard: Configuration  

In general, the main YAML configuration file is: `Dataset.yaml`  

You can modify the following as needed:  
```yaml
MODEL_NAME: HMOGPLV  
DATA_SPEC: Embed
EXPERIMENTTYPE: Missing_Tri_d4
```  

#### :game_die: EXPERIMENTTYPE Options 
```
Remove_Diag 			# for the bilingual experiment, - just diag-row removed
Extrapolate_n2 			# for the multilingual experiment
Missing_Triangle_1 		# for nanoGPTlarge(Embed) or nanoGPTsmall(Embed2)
Missing_Tri_d1 			# for nanoGPTlarge(Embed) or nanoGPTsmall(Embed2), dataset T1 # dx: feel free to change x, to have different train-test splits
Missing_Tri_d4 			# for nanoGPTlarge(Embed) or nanoGPTsmall(Embed2), dataset Tri # dx: feel free to change x, to have different train-test splits
Missing_Quad_d4_r3 		# for nanoGPTlarge(Embed) or nanoGPTsmall(Embed2), dataset Quad # dx_rx: feel free to change x, to have different train-test splits
Missing_One_replica_in_each_output # for the bilingual experiment, - diag-row and one learning curve per row
active_main 			# active learning experiment
active_random 			# active learning experiment
active_smallest 		# active learning experiment
active_largest 			# active learning experiment
Missing_One_replica_in_Whole
Train_test_in_each_replica
```  

#### :game_die: DATA_SPEC Options 
```
Multilingual_srcid_tgtx 		# Translation from source id into all tgt langs
Multilingual_srcx_tgtid 		# Translation into target id from all src langs
Embed 							# nanoGPTlarge dataset
Embed2 							# nanoGPTsmall dataset
Layer 							# exchanged hierarchies on the nanoGPTlarge dataset
Bilingual_modelmBart_metricchrf # Translation using mBart50 and metric ChrF
Bilingual_modelmBart_metricBLEU # Translation using mBart50 and metric BLEU
Bilingual_modelmTransformer_metricchrf # Translation using Transformer and metric ChrF
Bilingual_modelmTransformer_metricBLEU # Translation using Transformer and metric BLEU
```  

#### :game_die: Data Parameters 
```
NUM_REPLICATES: 6           # 4 for embed2, 5 for lang, 6 for embed/bilingual
NUM_OUTPUTS: 5              # 3 for embed2/lang, 5 for embed, 6 for bilingual
NUM_DATA_IN_REPLICATES: 11  # 20 for lang, 11 for embed, 17 for bilingual
```  

> :exclamation: Keep other settings as in the original MaGP model.

---  

#### :chart_with_upwards_trend: Results  

The results will be created in folder: `Results/MODEL_NAME/DATA_SPEC_EXPERIMENTTYPE`  

This folder includes:  
- A file called `error_metrics*` (metrics summary):  
  - **full:** entire learning curve,  
  - **partial:** last datapoints (3 for zero-shot, 9 for few-shot),  
  - **last:** final datapoint.  
  - Each run is recorded in one line.
  - Average over all runs, to get the average values.
- Plots showing predicted and ground truth learning curves, uncertainties; and predicted learning curves in z-score normalised and original domain.  

---  

## :pushpin: Experiments 
For MaGP and its most competitive baseline, set the following in the yaml.file:

- **DHGP:** `TRAINING_NUM_EACH_STEP: 10000`  
- **MaGP:** `TRAINING_NUM_EACH_STEP: 2000`  

---  

#### :large_blue_circle: Zero-Shot Prediction (NanoGPT Datasets)  

**Train-test split "Quad":**  
```bash
python main.py --cfg Dataset_Quad.yaml --runs_num 10
```  
Results: `Results/HMOGPLV/Embed_Missing_Quad_d4_r3/`

The results folder includes: 
- A file called `error_metrics*` (metrics summary as above)
- A plot "C_plot*" showing the learning curves used for training and the predicted learning curves.
- A folder `Embed` containing the numerical values of the full set of learningn curves (used for active learning later).
- Plots showing the prediction, ground truth and uncertainty of the model for the z-score normalised and original domain.
- Change the MODEL_NAME in the .yaml file to "DHGP" for the baseline model.

**Train-test split "Tri":**  
```bash
python main.py --cfg Dataset_Tri.yaml --runs_num 10
```  
Results folder will be created as above, according to this split.

**Train-test split "T1":**  
```bash
python main.py --cfg Dataset_T1.yaml --runs_num 10
```  
Results folder will be created as above, according to this split.

**Exchanged hierarchies:**  
Use corresponding `_exchHierarch.yaml` for each split.
Results folder will be created as above, according to this split.

**Scaling law plot (Figure 1):**  
To plot the scaling law using the full nanoGPT dataset:

```bash
cd Datasets/Dataset/init_data
python 0_get_wandb_data.py
```  

**To recreate datasets:**  
To create the nanoGPTlarge or nanoGPTsmall dataset as in our paper or using a different sampling of the original learning curves:
```bash
cd 0_get_data
python get_dict_hier.py
```  
The dataset will be saved in the folder `data`. For exchanged hierarchies, change number of layers and number of embedding parameters to be saved accordingly.

---  

#### :large_blue_circle: Zero-Shot Prediction (Bilingual Datasets)  

```bash
python main.py --cfg Dataset_modelmBart_metricbleu.yaml --runs_num 10
python main.py --cfg Dataset_modelmTransformer_metricbleu.yaml --runs_num 10
python main.py --cfg Dataset_modelmBart_metricchrf.yaml --runs_num 10
python main.py --cfg Dataset_modelmTransformer_metricchrf.yaml --runs_num 10
```  
Results folder will be created as above, according to these splits.

Exchanged hierarchies: modify `HMOGPLV/Load_data.py` lines 315–329:  
`"src_over_tgt"` → `"tgt_over_src"`  

Change the MODEL_NAME in the .yaml file to "DHGP" for the baseline model.


To regenerate bilingual learning curves:  
(In order to sample the learning curve coarser or finer or recreate the current dataset.)
```bash
cd Datasets/Dataset/bilingual/data_LC
python get_bilingual_data.py
python get_bilingual_data2.py  # exchanged hierarchy
```  
Output stored in `Datasets/Dataset/bilingual/data_LC/data`.

---  

#### :large_blue_circle: Few-Shot Prediction (Multilingual Datasets)  

Choose source/target language from: `'en', 'id', 'jv', 'ms', 'ta', 'tl'`.  

Example:
To extrapolate for the last 9 datapoints
```bash
python main.py --cfg Dataset_srcen_n9.yaml --runs_num 10
```  

Modify YAML:  
- `EXPERIMENTTYPE: Extrapolate_nx` for extrapolation depth  
- `DATA_SPEC: Multilingual_srcy_tgtx` for source language  
- Swap source/target: `Multilingual_srcx_tgten` / `Multilingual_srcx_tgty`  
	- More Examples:
		- To extrapolate the last x datapoints, change `EXPERIMENTTYPE: Extrapolate_nx` accordingly.
		- To translate from source language y, change `DATA_SPEC: Multilingual_srcy_tgtx` accordingly.
		- For exchanged hierarchies, change `DATA_SPEC: Multilingual_srcen_tgtx to Multilingual_srcx_tgten`.
		- To translate into target language x, change `DATA_SPEC: Multilingual_srcen_tgtx` accordingly.


To recreate the dataset:  
```bash
cd Datasets/Dataset/multilingual/LC_data
python rearrange_data.py
```  
The dataset will be created in the folder `data`.

---  

### :heavy_exclamation_mark: Active Learning Experiments  

**Main file to run for active learning:**  
```bash
python run_active.py
```  
Active learning folder: `0_get_data/AL/Active_main/`  
When running the experiments, this folder will contain:  
- `m_b_dicts`: m-v dictionaries  
- `SL_fit`: learning curves and scaling law plots  
- `uncertainty_dicts`: prediction uncertainties  

Results of MaGP/DHGP experiments are saved to:  
`Results/MODEL_NAME/DATA_SPEC_EXPERIMENTTYPE_query_number`  
(The number of queries can be set in run_active.py)\
This folder will contain files as above.

**Alternative query strategies:**  
```bash
python run_smallest.py   # smallest-first
python run_largest.py    # largest-first
python run_random.py     # random order
```  

**Analysis scripts:**  
(after all active learning experiments above are finished.)\
To compare AbC over query plots:
```bash
python 0_analyse_active_learning.py
```
To compare AbC over cost plots:
```bash
python 0_analyse_active_learning_cost.py
```
To plot query over compute cost:
```bash
python 0_calc_budget_query.py
```
To plot compute cost over query:
```bash
python 0_calc_budget_spend.py
```  

---  

## :closed_book: Citation  

```bibtex
@inproceedings{schram2025zero,
  title={Zero-Shot Performance Prediction for Probabilistic Scaling Laws},
  author={Schram, Viktoria and Hiller, Markus and Beck, Daniel and Cohn, Trevor},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={38},
  year={2025}
}
```
### :bulb: Acknowledgement

This repository is built on and inspired by:  
- MaGP: [HMOGP-LV GitHub](https://github.com/ChunchaoPeter/HMOGP-LV/tree/main)  
- Scaling law experiments: [Reconciling Kaplan-Chinchilla Scaling Laws](https://github.com/TeaPearce/Reconciling_Kaplan_Chinchilla_Scaling_Laws)  

We thank the authors for providing their code.