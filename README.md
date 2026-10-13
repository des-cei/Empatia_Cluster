# Empatia_Cluster

**Cluster-aware physiological emotion recognition for cloud–edge personalization.**

Empatia_Cluster contains research code for training and evaluating emotion-recognition models on groups of users with similar physiological characteristics. It combines physiological feature maps with cluster-specific CNN–LSTM models and includes experiments on unseen users, cross-cluster generalization, and supervised personalization.

This repository supports the **cluster-based cloud–edge adaptation framework in Chapter 4** of Junjiao Sun's doctoral thesis, *Toward Practical and Uncertainty-Aware Affective Computing with Wearable Physiological Signals*, and is associated with:

> Junjiao Sun, Laura Gutiérrez-Martín, Celia López-Ongil, José Miranda, Jorge Portilla, and Andrés Otero. **Solving the Cold-Start Problem for the Edge: Clustering and Adaptive Deep Learning for Emotion Detection.** *2025 Design, Automation & Test in Europe Conference (DATE)*, pp. 1–7, 2025. [DOI: 10.23919/DATE64628.2025.10992800](https://doi.org/10.23919/DATE64628.2025.10992800).

The feature-map and CNN–LSTM foundations are developed in the companion repositories [Empatia_DL](https://github.com/des-cei/Empatia_DL) and [Empatia_CNN-LSTM](https://github.com/des-cei/Empatia_CNN-LSTM).

> **Implementation status:** This is a collection of research scripts, primarily configured for WEMAC binary fear recognition. Cluster assignments are supplied through external CSV files. The feature cache, partition files, normalization log, and `FWN_main` dependency are not included. Local configuration and the compatibility adjustments below are required before training.

## Research context

Physiological responses vary across people. A single population-level model can overlook these differences, while training a separate model for every new user requires labeled personal data. The Chapter 4 framework addresses this cold-start problem through user-typology clusters:

1. **Cloud-side grouping:** Summarize each training user's physiological features and group users with similar response patterns.
2. **Cluster-specific pretraining:** Train a recognition model on the feature maps of each user group.
3. **Unlabeled user assignment:** Compare a new user's physiological representation with stored cluster representatives.
4. **Model selection:** Retrieve the corresponding pretrained model for local inference.

The thesis describes centroid-based clustering, selecting four clusters for WEMAC and two for WESAD. These are research configurations, not clustering settings automatically computed by this repository.

### Scope of the released code

| Component | Repository coverage |
| --- | --- |
| Physiological descriptors | BVP, GSR, and skin-temperature feature modules. |
| Normalization and maps | Cache loading, outlier processing, normalization interfaces, and map augmentation. |
| Cluster-aware recognition | Training and evaluation using participant assignments read from CSV files. |
| Personalization experiments | Cross-cluster testing and supervised fine-tuning variants. |
| Edge-oriented experiments | PyTorch checkpoint/quantization utilities and a separate TensorFlow/TFLite training script. |
| Clustering and cold-start assignment | The centroid-fitting and new-user assignment pipeline is not supplied; scripts consume its partition results. |
| Cloud–device integration | No complete model-serving, download, or device communication service is included. |

The active data loader uses WEMAC-style participant/video records and `ReportadaBinarizado` labels. A complete WESAD workflow is not included. The repository also does not provide the full feature-selection, distillation, and microcontroller deployment pipeline of Chapter 4's separate ultra-light framework.

**Cold-start selection and fine-tuning are different experiments.** The chapter's cold-start mechanism selects a model using unlabeled new-user data. Scripts that train on a subset of a new user's labels evaluate supervised adaptation and should be reported separately.

## Representation and models

### Parallel CNN–LSTM

The principal model is `ParallelCNNLSTMModel` in `Training/models/cnn_lstm.py`:

| Branch/stage | Implementation |
| --- | --- |
| CNN | Two `Conv1d` layers with 64 and 128 output channels, each followed by ReLU and max pooling; flatten and project to 128 dimensions. |
| LSTM | Two recurrent layers with hidden size 64 by default; project the final sequence output to 128 dimensions. |
| Fusion | Concatenate both branch embeddings into a 256-dimensional vector. |
| Classifier | Linear layer producing two logits for fear/non-fear classification. |

The branches process the same input in parallel. This implementation uses 1D convolutions, which should be distinguished from the chapter's description of the 2D CNN–LSTM architecture.

The model module also includes `SimpleLSTM` and an experimental `CNNLSTM_8Channel` that applies a 2D CNN before an LSTM. The latter has fixed internal dimensions and requires matching input shapes. Additional CNN architectures are available in `Training/models/`.

## Repository guide

| Directory | Contents |
| --- | --- |
| `Feature_extraction/` | Signal-feature functions and signal visualization. |
| `Feature_wise_Normalization/` | Feature-cache loading, normalization, and map generation. |
| `Abnormal_analyze/` | Outlier inspection, statistical feature analysis, and user-specific map experiments. |
| `Training/` | Cluster-specific training, cross-cluster tests, and fine-tuning experiments. |
| `Training/models/` | CNN–LSTM, LSTM, and CNN definitions. |
| `clustering_LOSO/` | CSV-driven subject-holdout and comparison experiments. |
| `Edge_training/` | Transfer, fine-tuning, checkpoint, and conversion experiments. |

## Setup

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/des-cei/Empatia_Cluster.git
cd Empatia_Cluster
python -m venv .venv
source .venv/bin/activate
export EMPATIA_CLUSTER_ROOT="$PWD"
export PYTHONPATH="$EMPATIA_CLUSTER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Install a compatible PyTorch/torchvision pair for your system. Other packages inferred from the source imports include:

```bash
python -m pip install numpy scipy pandas scikit-learn matplotlib \
    mat73 biosppy distfit skfeature-chappers
```

TensorFlow is additionally required for `Edge_training/train_general_WEMAC.py`. There is no pinned dependency file or documented, validated environment for this repository; the list above is an import-based starting point.

The main PyTorch loops explicitly cast inputs to `torch.cuda.FloatTensor`, so they require CUDA as written. To support CPU execution, replace those casts with device-aware conversion, such as `inputs.to(device=device, dtype=torch.float32)`.

## Required input files

### Feature cache and normalization log

The main experiments expect:

| Path relative to the repository root | Content |
| --- | --- |
| `Feature_wise_Normalization/Pack_all_data.json` | Nested participant → video → window dictionaries containing physiological descriptors and labels. |
| `Feature_wise_Normalization/log_data_normalization_IT06_knn.log` | Feature-wise transformation choices, including a completed `Over all Best N:` entry. |

Participants use identifiers such as `V001`; CSV participant IDs are integers converted to that format. Video keys must be numerically convertible. Each usable window must contain `ReportadaBinarizado`, and descriptor names must contain `bvp`, `gsr`, or `skt` to be selected by the loader. Keep feature ordering consistent across windows, participants, and the normalization log.

Obtain the source data separately under its access terms. This repository includes descriptor functions but no complete raw-recording-to-`Pack_all_data.json` exporter. Prepare the cache with a compatible upstream pipeline, such as the WEMAC preparation code in the companion repository, and verify recording-version and label compatibility.

### Cluster partition CSV

Most scripts in `Training/` read `partitions_traintest.csv` from the repository root through `Training/utils.py::get_clusters`.

The parser expects **semicolon-separated records**, with these first three fields:

| Field position | Meaning |
| --- | --- |
| 1 | Integer participant ID. |
| 2 | Integer cluster ID. |
| 3 | Integer partition/repetition ID. |

Each record must also contain a method marker, such as `train`, `test_method1`, or `test_method2`. The parser selects records by these text markers and returns dictionaries indexed by partition and cluster. An illustrative record layout is:

```text
1;1;0;train
2;1;0;test_method2
```

This illustrates the parser contract only; it is not an experimental partition. Preserve the assignment method and provenance of the real records. These marker names identify externally supplied assignments, not clustering algorithms implemented here. Several scripts assume cluster IDs 1–4 and partition indices that fit fixed result arrays.

### LOSO assignment CSV

The scripts in `clustering_LOSO/` expect a separate, **comma-separated** `clustering_LOSO.csv` in that directory, with named columns:

| Column | Meaning |
| --- | --- |
| `id_vol` | Participant ID for the row. |
| `id_cluster` | Assigned cluster ID. |
| `method` | Assignment role, including `method2_train` and `method2_test`. |
| `vol_test` | Held-out participant identifying the LOSO fold. |

For each fold, `get_clusters_loso` takes the cluster ID from the **last matching row** for `vol_test`, selects `method2_train` participants in that cluster, and tests on `vol_test`. Ensure the last matching row describes the held-out user's assigned cluster, and verify that this user is absent from training records.

The scripts write `test ACC` and `test F1` back into the input CSV. Work from a copy to preserve the original assignment file.

## Running experiments

The commands below assume the required files and dependency have been supplied and compatibility adjustments completed.

### Cluster-specific training

```bash
cd "$EMPATIA_CLUSTER_ROOT/Training"
python CNN_LSTM_approach1.py --lr 0.1
```

| Entry point | Experiment |
| --- | --- |
| `CNN_LSTM_approach1.py` | Use the explicit training/test participant lists from the assignment CSV. |
| `CNN_LSTM_approach2.py` | Combine the provided lists and randomly split participants approximately 80/20 within that group. |
| `Approach1_test_other_clusters.py`, `Approach2_test_other_clusters.py` | Cross-cluster evaluation experiments. |
| `Approach1_fine_tune.py`, `Approach2_fine_tune.py` | Cluster-model training followed by supervised new-user adaptation. |
| `CNN_LSTM_approach1_8Channel.py` | Alternative multi-channel architecture experiment. |

`approach1`/`approach2` control the **data split** and are distinct from the CSV's `test_method1`/`test_method2` assignment markers. Approach 2 reassigns membership and therefore does not preserve the CSV's original train/test split.

Edit the main block to select assignment markers, targets, clusters, strategies, and repetition counts. Start with one partition, one cluster, and one repetition before launching all combinations.

### Subject-holdout experiments

```bash
cd "$EMPATIA_CLUSTER_ROOT/clustering_LOSO"
python clustering_LOSO.py --lr 0.1
```

| Entry point | Behavior |
| --- | --- |
| `clustering_LOSO.py` | Train on the assigned cluster's training subjects and evaluate the held-out user. |
| `clustering_LOSO_fine.py` | Add the first 20% of generated held-out maps and labels to training; evaluate the remaining maps. This is supervised adaptation. |
| `clustering_baseline.py` | Train on the selected cluster and evaluate training-role subjects from other clusters. Despite its name, this is not an all-subject global-model baseline. |

Main CNN–LSTM defaults are cross-entropy loss, SGD with learning rate `0.1`, momentum `0.9`, weight decay `5e-4`, and cosine scheduling with `T_max=200`, for 100 epochs. Standard approach scripts use batch size 64; the LOSO scripts use 128. These PyTorch settings differ from the TensorFlow/Adam setup described in the thesis.

## Citation

If you use this code or the cluster-based personalization framework, please cite:

```bibtex
@inproceedings{sun2025coldstart,
  author    = {Sun, Junjiao and Gutiérrez-Martín, Laura and López-Ongil, Celia
               and Miranda, José and Portilla, Jorge and Otero, Andrés},
  title     = {Solving the Cold-Start Problem for the Edge: Clustering and Adaptive Deep Learning for Emotion Detection},
  booktitle = {2025 Design, Automation \& Test in Europe Conference (DATE)},
  year      = {2025},
  pages     = {1--7},
  doi       = {10.23919/DATE64628.2025.10992800}
}
```

Please also cite the relevant dataset and representation/model publications when using their data or methods.

## License

No repository-level license file is included. Contact the maintainers to clarify reuse and redistribution terms. Dataset access and reuse remain subject to the respective providers' terms.
