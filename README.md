# Stress Detection Pipeline Using the K-EmoPhone Dataset

This repository provides resources for independent reproducibility experiments focused on stress detection using the K-EmoPhone dataset. It includes a complete machine learning pipeline from data exploration to model evaluation and hyperparameter tuning. The environment is configured using Conda, with dependencies outlined in `environment.yaml`.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project leverages the K-EmoPhone dataset to develop and optimize models for detecting stress. The pipeline includes stages for data preprocessing, feature extraction, model training, evaluation, and hyperparameter tuning, enabling reproducibility and facilitating research in stress detection.

## Dataset

This project uses the K-EmoPhone dataset, which is described in the following publication:

Kim, H. R., Yang, J., Lee, J. H., & Kim, H. S. (2023). K-EmoPhone: Korean stress and emotion database using smartphone. Scientific data, 10(1), 351. <https://www.nature.com/articles/s41597-023-02248-2>

The dataset can be accessed via Zenodo. <https://doi.org/10.5281/zenodo.7606611>
## Project Structure

```bash
.
├── Funcs/
│   ├── Utility.py                           # Utility functions for data processing
├── EDA.ipynb                                # Exploratory Data Analysis
├── Feature_Extraction.ipynb                 # Feature Extraction notebook
├── HyperParameterTuning.ipynb               # Hyperparameter tuning notebook (here hyperparameter tuning is just a simple trial to test the impact of model complexity related hyperparameters)
├── Model Evaluation & Deeper Analysis.ipynb # Evaluation and deeper analysis                        
├── Model Training.ipynb                     # Model training procedures
├── Preprocessing.ipynb                      # Data preprocessing steps
├── Visualization_Overfitting.ipynb          # Overfitting visualization
│                                
└── environment.yaml                         # Conda environment configuration
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SteinPanyu/IndependentReproducibility.git
   cd Stress_Detection_D-1
   ```

2. Set up the Conda environment:

   ```bash
   conda env create -f environment.yaml
   conda activate sci-data
   ```

## Usage

The project is organized into Jupyter notebooks, each representing a stage of the machine learning pipeline. Follow the sequence below to reproduce the entire process:

## Pipeline Stages

1. **Exploratory Data Analysis (EDA)**
   - Notebook: `EDA.ipynb`
   - Purpose: Visualize and explore the data before preprocessing the sensor data.
   - Includes: Visualize the raw sensor data and labels

2. **Preprocessing**
   - Notebook: `Preprocessing.ipynb`
   - Purpose: Clean and transform the data for subsequent stages.
   - Includes: Handling missing data, normalization, and encoding.

3. **Feature Extraction**
   - Notebook: `Feature_Extraction.ipynb`
   - Purpose: Extract relevant features from the dataset.
   - Includes: Extraction of raw sensor data features.
   
   ![image](https://github.com/user-attachments/assets/4f9e59f8-f07b-4d22-b907-e831edd5fe34)


4. **Model Training**
   - Notebook: `Model Training.ipynb`
   - Purpose: Train machine learning models to classify stress levels.
   - Includes: Training procedures, model selection, and validation.

5. **Model Evaluation & Deeper Analysis**
   - Notebook: `Model Evaluation & Deeper Analysis.ipynb`
   - Purpose: Evaluate the model's performance with detailed analysis.
   - Includes: Metrics, interpretability, and areas for improvement.

6. **Overfitting Visualization**
   - Notebook: `Visualization_Overfitting.ipynb`
   - Purpose: Visualize and mitigate overfitting.
   - Includes: Visualize the training and testing performance plot.

7. **Hyperparameter Tuning (Overfitting Mitigation)**
   - Notebook: `HyperParameterTuning.ipynb`
   - Purpose: Optimize model complexity related parameters to mitigate overfitting.
   - Includes: HyperOPT for model complexity related hyperparameters

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- K-EmoPhone Dataset (Kang et al., 2023) for providing the data used in this project.
- scikit-learn and pandas libraries for data processing and model building.
