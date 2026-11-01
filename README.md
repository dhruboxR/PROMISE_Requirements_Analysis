# PROMISE+ Requirements Analysis

**Exploratory Data Analysis and Machine Learning Research for Software Requirements Engineering**

This repository contains research work focused on analyzing and preparing the PROMISE+ software requirements dataset for machine learning-based requirements classification and further investigation of requirements quality and inconsistency.

The project follows a structured research workflow, beginning with dataset validation and exploratory data analysis (EDA), followed by preprocessing, model development, and evaluation.

## 1. Research Objectives

The primary objectives of this research are:

- Explore and analyze the PROMISE+ requirements dataset.
- Identify data-quality issues, duplicate requirements, and potential annotation inconsistencies.
- Examine requirement class distribution and class imbalance.
- Analyze linguistic characteristics and requirement lengths.
- Prepare a reproducible dataset-processing workflow.
- Develop and evaluate machine learning approaches for software requirements classification.
- Investigate approaches for detecting potential inconsistencies in software requirements.

**Research status:** Phase 1, Dataset Exploration, has been completed. Further preprocessing and model development are planned.

## 2. Dataset Information

The project uses the PROMISE+ software requirements dataset, provided in ARFF format.

| Property | Description |
|---|---|
| Dataset | PROMISE+ |
| Format | ARFF |
| Total requirements | 3,677 |
| Unique projects | 88 |
| Dataset attributes | 3 |
| Requirement categories | 12 |
| Missing values | 0 |
| Blank requirement texts | 0 |

### Dataset Attributes

| Attribute | Description |
|---|---|
| `ProjectID` | Identifier of the software project |
| `RequirementText` | Natural-language software requirement |
| `_class_` | Classification label assigned to the requirement |

The dataset contains the following requirement categories:

`F`, `A`, `L`, `LF`, `MN`, `O`, `PE`, `SC`, `SE`, `US`, `FT`, `PO`

The original dataset is preserved in `dataset/Promise+.arff`. A structurally repaired copy is generated for analysis.

## 3. Repository Structure

```text
PROMISE_Requirements_Analysis/
├── dataset
│   ├── Lista dos tipos de software.xlsx
│   └── Promise+.arff
├── Docs
│   └── Thesis_Intro+LiteratureReview.pdf
├── promise_exploration_phase_1
│   ├── code_outputs_phase1.txt
│   ├── outputs
│   ├── README.md
│   └── scripts
├── promise_preprocessing_phase_2
│   ├── code_outputs_phase2.txt
│   ├── outputs
│   ├── README.md
│   └── scripts
├── README.md
├── requirements.lock.txt
└── requirements.txt
```

## 4. Technologies and Libraries

The project uses Python for data analysis and visualization.

Main technologies include:

- Python
- pandas
- NumPy
- liac-arff
- Matplotlib
- Seaborn

Exact project dependencies are documented in:

- `promise_exploration_phase_1/requirements.txt`
- `promise_exploration_phase_1/requirements.lock.txt`

The requirements files are the authoritative dependency lists.

## 5. Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/dhruboxR/PROMISE_Requirements_Analysis.git
cd PROMISE_Requirements_Analysis
```

### Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### Install Dependencies

```bash
python -m pip install -r promise_exploration_phase_1/requirements.txt
```

For an environment matching the recorded package versions, use `requirements.lock.txt` instead, subject to Python-version and platform compatibility.


## 6. Research Workflow

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | Dataset validation and exploratory analysis | Completed |
| Phase 2 | Dataset preprocessing and splitting | Completed |
| Phase 3 | Machine learning model development | Planned |
| Phase 4 | Model evaluation and comparative analysis | Planned |
| Phase 5 | Requirements inconsistency investigation | Planned |

Future phases will address duplicate handling, leakage-controlled evaluation, classification methods, and approaches to inconsistency detection.

## 7. Research Considerations

Several observations from Phase 1 will guide future experiments:

- The original dataset requires documented ARFF formatting corrections.
- The class distribution is substantially imbalanced.
- Some requirement texts are repeated within or across projects.
- Four normalized-text groups have conflicting classification labels.
- Project-based splitting alone may not prevent identical requirements from appearing across different partitions.
- The available classification labels do not directly establish whether requirements semantically contradict one another.

The research will consider these limitations during preprocessing and evaluation.

## 8. Documentation

The introduction and literature review are stored in:

`Docs/Thesis_Intro+LiteratureReview.pdf`

Further methodology and experimental documentation will be added as the project develops.

---

**Note:** This repository is intended for academic research. Dataset ownership and redistribution permissions remain with their respective rights holders. Any public redistribution should follow the original dataset's applicable licensing and attribution requirements.
