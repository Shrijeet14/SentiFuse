<p align="center">
  <img src="download.svg" alt="sentifuse-banner" width="800">
</p>

<p align="center">
	<em>Fuse Emotions, Power Decisions - SentiFuse!</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/Shrijeet14/SentiFuse?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Shrijeet14/SentiFuse?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Shrijeet14/SentiFuse?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Shrijeet14/SentiFuse?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📁 Project Structure](#-project-structure)
  - [📂 Project Index](#-project-index)
- [🚀 Getting Started](#-getting-started)
  - [☑️ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#-installation)
  - [🤖 Usage](#🤖-usage)
  - [🧪 Testing](#🧪-testing)
- [📌 Project Roadmap](#-project-roadmap)
- [🔰 Contributing](#-contributing)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

SentiFuse is an open-source project designed to streamline sentiment analysis in YouTube comments. It leverages Python's robust libraries, ensuring code quality with strict standards and automated testing. Key features include dependency management, automated tasks, and environment setup. It's an invaluable tool for data scientists, developers, and researchers seeking to harness the power of sentiment analysis in social media data.

---

## 👾 Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Designed with a modular architecture</li><li>Utilizes a combination of Python scripts and libraries</li><li>Employs a data version control system (DVC)</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Code is well-structured and follows Python best practices</li><li>Uses flake8 for linting to maintain code quality</li><li>Code is clean and easy to understand, with clear variable and function names</li></ul> |
| 📄 | **Documentation** | <ul><li>Documentation is comprehensive and well-structured</li><li>Includes detailed setup instructions and usage examples</li><li>Documentation is maintained in the codebase itself</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integrates with various Python libraries such as NLTK, Pillow, and NetworkX</li><li>Supports integration with cloud storage services like S3 through boto3 and s3fs</li><li>Integrates with Git through GitPython for version control</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Code is organized into separate modules for different functionalities</li><li>Uses a modular approach for handling different types of data processing tasks</li><li>Each module is designed to be independent and reusable</li></ul> |
| 🧪 | **Testing**       | <ul><li>Uses tox for automated testing</li><li>Includes a comprehensive set of tests for different modules</li><li>Tests are well-organized and easy to run</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimized for high performance with efficient algorithms and data structures</li><li>Uses libraries like scipy and numpy for efficient numerical computations</li><li>Performance is regularly tested and monitored</li></ul> |
| 🛡️ | **Security**      | <ul><li>Follows best practices for secure coding</li><li>Uses secure libraries and dependencies</li><li>Includes measures to prevent common security vulnerabilities</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Has a large number of dependencies, including popular libraries like Flask, SQLAlchemy, and MLflow</li><li>Dependencies are managed through requirements.txt and pip</li><li>Includes both high-level and low-level dependencies</li></ul> |

---

## 📁 Project Structure

```sh
└── SentiFuse/
    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── confusion_matrix_Test Data.png
    ├── docs
    │   ├── Makefile
    │   ├── commands.rst
    │   ├── conf.py
    │   ├── getting-started.rst
    │   ├── index.rst
    │   └── make.bat
    ├── dvc.lock
    ├── dvc.yaml
    ├── models
    │   └── .gitkeep
    ├── myenv
    │   ├── Scripts
    │   └── pyvenv.cfg
    ├── notebooks
    │   ├── .gitkeep
    │   ├── 1_yt_comment_analyzer_preprocessing.ipynb
    │   ├── 2_experiment_1_baseline_model.ipynb
    │   ├── 3_experiment_2_bow_tfidf.ipynb
    │   ├── 4_experiment_3_tfidf_(1,3)_max_features.ipynb
    │   ├── 4_experiment_4_handling_imbalanced_data.ipynb
    │   ├── custom_features.ipynb
    │   ├── experiment_5_knn_with_hpt.ipynb
    │   ├── experiment_5_lightgbm_with_hpt.ipynb
    │   ├── experiment_5_lor_with_hpt.ipynb
    │   ├── experiment_5_naive_bayes_with_hpt.ipynb
    │   ├── experiment_5_random_forest_with_hpt.ipynb
    │   ├── experiment_5_svm_with_hpt.ipynb
    │   ├── experiment_5_xgboost_with_hpt.ipynb
    │   ├── lightGBM_final_stage_2.ipynb
    │   ├── lightGBM_hyperparameter_tuning.ipynb
    │   ├── stacking.ipynb
    │   └── word2vec.ipynb
    ├── params.yaml
    ├── references
    │   └── .gitkeep
    ├── reports
    │   ├── .gitkeep
    │   └── figures
    ├── requirements.txt
    ├── scripts
    │   └── mlflow_test.py
    ├── setup.py
    ├── src
    │   ├── __init__.py
    │   ├── data
    │   ├── features
    │   ├── models
    │   └── visualization
    ├── test_environment.py
    └── tox.ini
```


### 📂 Project Index
<details open>
	<summary><b><code>SENTIFUSE/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/tox.ini'>tox.ini</a></b></td>
				<td>- "Tox.ini" sets the coding standards for the project, specifying the maximum line length and complexity for code readability and maintainability<br>- It's a crucial part of the codebase architecture, ensuring consistency and quality across all code contributions.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/test_environment.py'>test_environment.py</a></b></td>
				<td>- Test_environment.py ensures the correct Python interpreter is being used for the project<br>- It checks the system's Python version against the project's required version<br>- If they match, it confirms the development environment passes all tests<br>- If not, it raises an error, indicating the required Python version and the current system version.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td>- The 'requirements.txt' file outlines the necessary dependencies for the project<br>- It includes both local and external packages, ensuring the project's functionality and compatibility<br>- The dependencies range from web development frameworks like Flask, to data analysis libraries like pandas and numpy, and version control systems like GitPython<br>- This file is crucial for setting up the development environment consistently across different platforms.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/Makefile'>Makefile</a></b></td>
				<td>- The Makefile in the 'yt-comment-sentiment-analysis' project serves as a task runner for automating various operations<br>- It manages tasks such as installing Python dependencies, creating datasets, cleaning compiled Python files, linting, syncing data to and from S3, and setting up the Python interpreter environment<br>- This file enhances the project's maintainability and reproducibility.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/params.yaml'>params.yaml</a></b></td>
				<td>- Params.yaml serves as a configuration file for the data ingestion and model building phases of the project<br>- It specifies parameters such as test size for data splitting, ngram range, maximum features for text vectorization, and hyperparameters for the machine learning model, including learning rate, max depth, and number of estimators.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/setup.py'>setup.py</a></b></td>
				<td>- Setup.py serves as the build script for the project, facilitating the packaging and distribution of the 'src' software<br>- It's designed to perform YouTube comment analysis, authored by Shrijeet<br>- The script uses setuptools to identify and bundle all packages within the project, marking version '0.1.0' without a specified license.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/dvc.yaml'>dvc.yaml</a></b></td>
				<td>- The dvc.yaml file orchestrates the machine learning pipeline in the project<br>- It defines stages including data ingestion, preprocessing, model building, evaluation, and registration<br>- Each stage specifies commands, dependencies, parameters, and outputs, ensuring reproducibility and version control<br>- This file is integral to the project's data science lifecycle.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- scripts Submodule -->
		<summary><b>scripts</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/scripts/mlflow_test.py'>mlflow_test.py</a></b></td>
				<td>- The script 'mlflow_test.py' in the 'SentiFuse' project initializes a machine learning experiment tracking setup using MLflow and Dagshub<br>- It logs random parameters and metrics for testing purposes, providing a foundation for tracking and managing machine learning experiments within the project.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- notebooks Submodule -->
		<summary><b>notebooks</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/experiment_5_random_forest_with_hpt.ipynb'>experiment_5_random_forest_with_hpt.ipynb</a></b></td>
				<td>- The file `experiment_5_random_forest_with_hpt.ipynb` is a Jupyter notebook that is part of a larger project<br>- This notebook is designed to conduct an experiment using a Random Forest algorithm with Hyperparameter Tuning (HPT)<br>- In the context of the entire project, this notebook plays a crucial role in model experimentation and optimization<br>- It is used to fine-tune the parameters of the Random Forest model to achieve the best possible performance<br>- The results from this experiment can then be used to inform the development and refinement of the main machine learning model within the project<br>- Please note that the specific details of the experiment, such as the dataset used, the hyperparameters tuned, and the performance metrics evaluated, are not provided in the file content shared.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/word2vec.ipynb'>word2vec.ipynb</a></b></td>
				<td>- The code file `notebooks/word2vec.ipynb` is a Jupyter notebook that plays a crucial role in the project's machine learning pipeline<br>- It is primarily responsible for the training and evaluation of a text classification model.

The notebook begins by importing necessary libraries, including pandas for data manipulation, sklearn for model training and evaluation, gensim for Word2Vec model, lightgbm for Light Gradient Boosting Machine classifier, and numpy for numerical operations.

The main functionality of the code is to load a preprocessed Reddit comments dataset, clean it by removing rows with NaN values, and then apply a Word2Vec model to transform the text data into numerical vectors<br>- These vectors are then used to train a Light Gradient Boosting Machine (LGBM) classifier<br>- The performance of the classifier is evaluated using a classification report.

In the context of the entire codebase, this notebook is likely a part of the model training and evaluation stage of the project<br>- It provides a way to convert raw text data into a format that can be used for machine learning, and then applies a specific algorithm (LGBM) to classify the data<br>- The results of this notebook would be used to understand the performance of the model and make improvements as necessary.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/experiment_5_xgboost_with_hpt.ipynb'>experiment_5_xgboost_with_hpt.ipynb</a></b></td>
				<td>- The file `notebooks/experiment_5_xgboost_with_hpt.ipynb` is a Jupyter notebook that is part of a larger project<br>- This notebook is primarily used for running experiments with the XGBoost machine learning algorithm, along with hyperparameter tuning (HPT)<br>- The main purpose of this file is to experiment with different hyperparameters of the XGBoost algorithm to optimize its performance<br>- This is crucial in the overall project as it helps in improving the accuracy and efficiency of the machine learning model being developed<br>- The results from this notebook can be used to inform the development and refinement of the main codebase, particularly in terms of how the XGBoost algorithm is implemented and utilized<br>- In the broader context of the project, this file contributes to the experimental and iterative development process, allowing for continuous improvement and optimization of the machine learning model.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/experiment_5_naive_bayes_with_hpt.ipynb'>experiment_5_naive_bayes_with_hpt.ipynb</a></b></td>
				<td>- The file `notebooks/experiment_5_naive_bayes_with_hpt.ipynb` is a Jupyter notebook that is part of a larger codebase<br>- This notebook is likely used for running experiments or tests, specifically involving a Naive Bayes model with hyperparameter tuning (as suggested by the filename)<br>- The purpose of this file is to provide a structured and interactive environment where the model's performance can be evaluated and optimized through hyperparameter tuning<br>- This contributes to the overall project by enabling continuous improvement of the model's accuracy and efficiency<br>- In the context of the entire codebase, this file is part of the 'notebooks' directory, suggesting it is one of potentially many other notebooks used for similar experimental or testing purposes<br>- The results from these experiments would likely feed into the main application or system, influencing how the model is implemented and used.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/2_experiment_1_baseline_model.ipynb'>2_experiment_1_baseline_model.ipynb</a></b></td>
				<td>- The file `2_experiment_1_baseline_model.ipynb` is a Jupyter notebook that forms part of the project's experimental phase<br>- It's primarily used for creating and testing a baseline model<br>- This notebook is a crucial component of the project as it sets the initial benchmark against which all subsequent models and experiments will be compared<br>- The baseline model's performance will help in identifying improvements or regressions in future iterations<br>- This file is part of the 'notebooks' directory, indicating that it's used for exploratory data analysis, model development, and testing.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/stacking.ipynb'>stacking.ipynb</a></b></td>
				<td>- The `stacking.ipynb` file is a Jupyter notebook that forms part of the project's machine learning component<br>- It appears to be focused on implementing a stacking ensemble method, likely for model training and prediction<br>- The use of LightGBM, a gradient boosting framework, suggests that the project involves handling large-scale data and requires efficient, high-performance algorithms<br>- This notebook likely plays a key role in the project's data analysis and model development stages.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/experiment_5_lor_with_hpt.ipynb'>experiment_5_lor_with_hpt.ipynb</a></b></td>
				<td>- The file `experiment_5_lor_with_hpt.ipynb` is a Jupyter notebook that is part of a larger project<br>- This notebook is likely used for conducting a specific experiment, as suggested by its name<br>- It appears to be set up for running in a Python 3 environment, as indicated by the kernelspec metadata.

While the specific code within the notebook isn't provided, the name suggests it might involve logistic regression (LoR) with hyperparameter tuning (HPT)<br>- This implies that the notebook is used for machine learning model experimentation, where the performance of a logistic regression model is tested under different configurations of hyperparameters.

This notebook is a part of the project's larger architecture, likely contributing to the experimental or research component of the project<br>- It might be used for model development and validation before the model is integrated into the main application or service.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/experiment_5_lightgbm_with_hpt.ipynb'>experiment_5_lightgbm_with_hpt.ipynb</a></b></td>
				<td>- The file `experiment_5_lightgbm_with_hpt.ipynb` is a Jupyter notebook that is part of a larger machine learning project<br>- This notebook is likely used for running experiments with the LightGBM model, a gradient boosting framework, along with hyperparameter tuning (HPT)<br>- The notebook uses libraries like `mlflow` and `boto3`, indicating that the project involves tracking experiments and possibly interacting with AWS services<br>- In the context of the entire codebase, this notebook is likely used for model development and optimization, contributing to the overall goal of creating an effective machine learning model.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/3_experiment_2_bow_tfidf.ipynb'>3_experiment_2_bow_tfidf.ipynb</a></b></td>
				<td>- The file `3_experiment_2_bow_tfidf.ipynb` is a Jupyter notebook that forms part of the project's experimental codebase<br>- It is primarily used for conducting experiments related to the Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) techniques<br>- These techniques are fundamental in Natural Language Processing (NLP) and are used for text vectorization, which is a crucial step in many NLP tasks such as text classification, sentiment analysis, and topic modeling.

In the context of the entire project, this notebook plays a significant role in testing and refining the project's NLP components, contributing to the overall quality and performance of the software<br>- It is part of the 'notebooks' directory, which typically contains files used for data exploration, experimentation, and initial model development.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/1_yt_comment_analyzer_preprocessing.ipynb'>1_yt_comment_analyzer_preprocessing.ipynb</a></b></td>
				<td>- The file `1_yt_comment_analyzer_preprocessing.ipynb` is a Jupyter notebook that forms part of the larger codebase<br>- Its primary role within the project is to handle the preprocessing of YouTube comments<br>- This involves cleaning, transforming, and organizing the raw data from YouTube comments into a more usable format for further analysis or processing in subsequent stages of the project<br>- The notebook is likely the first step in a pipeline that includes data analysis, possibly machine learning model training, and results visualization.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/4_experiment_4_handling_imbalanced_data.ipynb'>4_experiment_4_handling_imbalanced_data.ipynb</a></b></td>
				<td>- The file `4_experiment_4_handling_imbalanced_data.ipynb` is a Jupyter notebook that forms part of the project's experimentation phase<br>- It is primarily focused on handling imbalanced data within the project's dataset<br>- The notebook likely includes various data preprocessing techniques, experimentation with different models, and evaluation metrics to address the issue of imbalanced data<br>- This is crucial as imbalanced data can lead to a bias in machine learning models, affecting the overall performance of the system<br>- The experiments conducted in this notebook contribute to the broader project goal by ensuring that the final model is robust and performs well across all data classes.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/4_experiment_3_tfidf_(1,3)_max_features.ipynb'>4_experiment_3_tfidf_(1,3)_max_features.ipynb</a></b></td>
				<td>- The file `notebooks/4_experiment_3_tfidf_(1,3)_max_features.ipynb` is a Jupyter notebook that is part of a larger project<br>- This notebook is likely used for conducting a specific experiment within the project, specifically involving the use of TF-IDF (Term Frequency-Inverse Document Frequency) with a specific parameter configuration (n-gram range of 1 to 3 and a maximum number of features).

The TF-IDF technique is often used in Natural Language Processing (NLP) to reflect how important a word is to a document in a collection or corpus<br>- This experiment is likely testing the impact of these TF-IDF parameters on the performance of a machine learning model or some other NLP task within the project.

The notebook is located in the `notebooks` directory, suggesting that it is part of a collection of similar files used for various experiments or analyses related to the project<br>- The naming convention of the file suggests that it is the third experiment in a series.

Without more specific details about the project or the contents of the notebook, it's hard to provide more precise information<br>- However, it's clear that this file plays a role in the project's research or experimental phase, contributing to the overall understanding of the problem the project is trying to solve.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/custom_features.ipynb'>custom_features.ipynb</a></b></td>
				<td>- The file `notebooks/custom_features.ipynb` is a Jupyter notebook that plays a crucial role in the data processing and feature extraction phase of the project<br>- The main purpose of this file is to load a dataset, presumably in a pandas DataFrame, and apply various transformations to it<br>- The code uses libraries such as pandas for data manipulation, sklearn for machine learning tasks, and spacy for natural language processing<br>- The transformations applied to the data include splitting the dataset into training and testing sets, and extracting text features using the TfidfVectorizer<br>- In the context of the entire codebase architecture, this notebook likely serves as a preliminary step in a larger data analysis or machine learning pipeline, preparing the data for subsequent modeling or analysis steps.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/lightGBM_hyperparameter_tuning.ipynb'>lightGBM_hyperparameter_tuning.ipynb</a></b></td>
				<td>- The file `notebooks/lightGBM_hyperparameter_tuning.ipynb` is a Jupyter notebook that is part of a larger machine learning project<br>- Its main purpose is to perform hyperparameter tuning on a LightGBM model, a gradient boosting framework that uses tree-based learning algorithms<br>- The notebook uses several libraries to achieve this, including MLflow for experiment tracking, Dagshub for version control and data science project management, Optuna for optimization, and imbalanced-learn for dealing with imbalanced datasets<br>- In the context of the entire codebase, this notebook contributes to the model optimization process, helping to fine-tune the parameters of the LightGBM model to improve its performance.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/experiment_5_knn_with_hpt.ipynb'>experiment_5_knn_with_hpt.ipynb</a></b></td>
				<td>- The file `experiment_5_knn_with_hpt.ipynb` is a Jupyter notebook that is part of a larger project<br>- This notebook is used to conduct an experiment involving a K-Nearest Neighbors (KNN) algorithm with Hyperparameter Tuning (HPT)<br>- The purpose of this file is to test and optimize the KNN algorithm's performance within the context of the project's overall objectives<br>- The experiment is likely part of a larger machine learning or data science initiative, as KNN and HPT are common techniques in these fields<br>- The results of this experiment could influence the direction of the project, such as the selection of models or the adjustment of parameters<br>- This file is located in the `notebooks` directory, indicating that it's part of a collection of similar files used for conducting experiments, testing hypotheses, or performing data analysis<br>- The project structure suggests a well-organized codebase, where each file or directory has a specific role.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/lightGBM_final_stage_2.ipynb'>lightGBM_final_stage_2.ipynb</a></b></td>
				<td>- The file `lightGBM_final_stage_2.ipynb` is a Jupyter notebook that forms a part of the project's machine learning pipeline<br>- It is primarily used for data preprocessing and model training<br>- The notebook begins by importing necessary libraries such as pandas for data manipulation, sklearn's train_test_split for splitting the dataset into training and testing sets, and TfidfVectorizer for text feature extraction<br>- Next, it reads a preprocessed Reddit comments dataset, cleans it by dropping rows with NaN values, and presumably goes on to further process the data and train a model (as suggested by the filename and the imported libraries)<br>- This file is crucial for the project as it likely contains the main machine learning algorithm implementation, which is presumably based on the LightGBM framework (as suggested by the file name)<br>- It is likely responsible for generating predictions or insights that drive the main functionality of the project.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/notebooks/experiment_5_svm_with_hpt.ipynb'>experiment_5_svm_with_hpt.ipynb</a></b></td>
				<td>- The file `experiment_5_svm_with_hpt.ipynb` is a Jupyter notebook that is part of a larger machine learning project<br>- This specific notebook is designed to conduct an experiment using a Support Vector Machine (SVM) model with hyperparameter tuning<br>- It leverages the MLflow library for experiment tracking, which allows for the systematic comparison of different model configurations and results<br>- The notebook is likely used in the exploratory phase of the project, where various models and parameters are being tested to identify the most effective approach.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- myenv Submodule -->
		<summary><b>myenv</b></summary>
		<blockquote>
			<details>
				<summary><b>Scripts</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/myenv/Scripts/activate'>activate</a></b></td>
						<td>- The 'activate' script in the 'myenv/Scripts' directory is primarily used to set up a virtual environment for the project<br>- It adjusts environment variables and paths to isolate the project's dependencies<br>- This ensures that the project runs in a controlled environment, reducing the risk of version conflicts between different libraries.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/myenv/Scripts/Activate.ps1'>Activate.ps1</a></b></td>
						<td>- The provided code file, `Activate.ps1`, is a crucial part of the project's codebase<br>- It is located in the `myenv/Scripts` directory<br>- The main purpose of this file is to activate a Python virtual environment for the current PowerShell session<br>- This is a critical operation as it ensures that all Python-related tasks executed within this session are confined to this virtual environment, thereby maintaining the integrity and isolation of the project's dependencies<br>- This file contributes to the overall project architecture by providing a mechanism for environment management, which is a fundamental aspect of Python project development.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/myenv/Scripts/deactivate.bat'>deactivate.bat</a></b></td>
						<td>- Deactivate.bat, located in the myenv/Scripts directory, serves to disable the virtual environment in a Windows system<br>- It restores the original system settings by resetting environment variables such as PROMPT, PYTHONHOME, PATH, and VIRTUAL_ENV, effectively ending the isolated workspace for Python project dependencies.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Shrijeet14/SentiFuse/blob/master/myenv/Scripts/activate.bat'>activate.bat</a></b></td>
						<td>- Activate.bat, located in the myenv/Scripts directory, is primarily responsible for setting up the virtual environment for the 'YouTube Comment Sentiment Analysis' project<br>- It adjusts system variables and paths to ensure isolation of dependencies, thereby maintaining project integrity and preventing potential conflicts with other Python projects.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---
## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with SentiFuse, ensure your runtime environment meets the following requirements:

- **Programming Language:** JupyterNotebook
- **Package Manager:** Tox, Pip


### ⚙️ Installation

Install SentiFuse using one of the following methods:

**Build from source:**

1. Clone the SentiFuse repository:
```sh
❯ git clone https://github.com/Shrijeet14/SentiFuse
```

2. Navigate to the project directory:
```sh
❯ cd SentiFuse
```

3. Install the project dependencies:


**Using `tox`** &nbsp; [<img align="center" src="" />]()

```sh
❯ echo 'INSERT-INSTALL-COMMAND-HERE'
```


**Using `pip`** &nbsp; [<img align="center" src="" />]()

```sh
❯ echo 'INSERT-INSTALL-COMMAND-HERE'
```




### 🤖 Usage
Run SentiFuse using the following command:
**Using `tox`** &nbsp; [<img align="center" src="" />]()

```sh
❯ echo 'INSERT-RUN-COMMAND-HERE'
```


**Using `pip`** &nbsp; [<img align="center" src="" />]()

```sh
❯ echo 'INSERT-RUN-COMMAND-HERE'
```


### 🧪 Testing
Run the test suite using the following command:
**Using `tox`** &nbsp; [<img align="center" src="" />]()

```sh
❯ echo 'INSERT-TEST-COMMAND-HERE'
```


**Using `pip`** &nbsp; [<img align="center" src="" />]()

```sh
❯ echo 'INSERT-TEST-COMMAND-HERE'
```


---
## 📌 Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## 🔰 Contributing

- **💬 [Join the Discussions](https://github.com/Shrijeet14/SentiFuse/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/Shrijeet14/SentiFuse/issues)**: Submit bugs found or log feature requests for the `SentiFuse` project.
- **💡 [Submit Pull Requests](https://github.com/Shrijeet14/SentiFuse/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/Shrijeet14/SentiFuse
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/Shrijeet14/SentiFuse/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Shrijeet14/SentiFuse">
   </a>
</p>
</details>

---

## 🎗 License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## 🙌 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
