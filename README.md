ml-ops-project-s22
==============================

This is an exam project for the DTU course Machine Learning Operations – 02476.

This project is about classifying IMDb reviews as negative or positive using the following <a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews">dataset</a> from kaggle.

The dataset contains of 49582 reviews in the form of texts. The reviews are binary annotated with either negative or positive. The dataset is divided into a traingset, testset and a validationset.

- training: 60 % 
- testing: 30 % 
- validation: 10 % 

The model used in this project is the pre-trained transformer model Electra from <a href="https://huggingface.co/docs/transformers/model_doc/electra">huggingface</a>.

We obtained a macro F1 score of 0.75

The following plot shows the obtained confusion matrix from the testset.

![CM](cm.png)

The project uses `cookie-cutter` structure to standardize the repository structure. Furthermore the project uses Data Version Control `DVC` to keep track of large files when stored on remote locations. Due to the computational cost the model is trained using `Google Cloud Project`. `WandB` is used for experimental logging. The project is `pep8` compliant.

Authors, <br>
Kelvin Foster -s174210<br>
Lasse Hansen -s154446<br>
Magnus Mortensen -s164814<br>

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>
⊂_ヽ<br>
　 ＼＼<br>
　　 ＼( ͡° ͜ʖ ͡°)<br>
　　　 >　⌒ヽ<br>
　　　/ 　 へ＼<br>
　　 /　　/　＼＼<br>
　　 ﾚ　ノ　　 ヽ_つ<br>
　　/　/<br>
　 /　/|<br>
　(　(ヽ<br>
　|　|、＼<br>
　| 丿 ＼ ⌒)<br>
　| |　　) /<br>
ノ )　　Lﾉ<br>
(_／<br>
</small></p>
