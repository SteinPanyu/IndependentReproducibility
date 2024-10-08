# Stress Detection Using the K-EmoPhone Dataset

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

Kim, H. R., Yang, J., Lee, J. H., & Kim, H. S. (2023). K-EmoPhone: Korean stress and emotion database using smartphone. Scientific data, 10(1), 351. <https://doi.org/10.1038/s41597-023-02249-1>

## Project Structure

```bash
.
├── EDA.ipynb                                # Exploratory Data Analysis
├── Feature_Extraction.ipynb                 # Feature Extraction notebook
├── Funcs/
│   ├── Utility.py                           # Utility functions for data processing
├── HyperParameterTuning.ipynb               # Hyperparameter tuning notebook
├── Model Evaluation & Deeper Analysis.ipynb # Evaluation and deeper analysis                        
├── Model Training.ipynb                     # Model training procedures
├── Preprocessing.ipynb                      # Data preprocessing steps
├── Visualization_Overfitting.ipynb          # Overfitting visualization and mitigation
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
   - Purpose: Visualize and explore the data to identify patterns indicating stress.
   - Includes: Both audio and text data insights.

2. **Preprocessing**
   - Notebook: `Preprocessing.ipynb`
   - Purpose: Clean and transform the data for subsequent stages.
   - Includes: Handling missing data, normalization, and encoding.

3. **Feature Extraction**
   - Notebook: `Feature_Extraction.ipynb`
   - Purpose: Extract relevant features from the dataset.
   - Includes: Extraction of pitch, intensity, and text-based markers.

4. **Model Training**
   - Notebook: `Model Training.ipynb`
   - Purpose: Train machine learning models to classify stress levels.
   - Includes: Training procedures, model selection, and validation.

5. **Hyperparameter Tuning**
   - Notebook: `HyperParameterTuning.ipynb`
   - Purpose: Optimize model parameters to improve performance.
   - Includes: Grid search and cross-validation techniques.

6. **Model Evaluation & Deeper Analysis**
   - Notebook: `Model Evaluation & Deeper Analysis.ipynb`
   - Purpose: Evaluate the model's performance with detailed analysis.
   - Includes: Metrics, interpretability, and areas for improvement.

7. **Overfitting Visualization**
   - Notebook: `Visualization_Overfitting.ipynb`
   - Purpose: Visualize and mitigate overfitting.
   - Includes: Regularization and model complexity adjustments.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- K-EmoPhone Dataset (Kim et al., 2023) for providing the data used in this project.
- scikit-learn and pandas libraries for data processing and model building.
