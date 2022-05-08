## Assignment 3

### Description
+ data folder:
  + Contains the hindi lexicon dataset
  + But since the dataset is big we have not put that dataset on github

+ prediction_attention folder:
  + contains predictions of test dataset using the best model with attention

+ prediction_vanilla folder:
  + contains predictions of test dataset using the best model with no attention

+ scripts:
  + ipynb scripts that we used for testing and generating plots

+ Codes:
  + dataset.py:
    + Read train, validation, test data from the data folder
  + model.py
    + Contains Attention, Encoder, Decoder class
  + model_runner.py
    + Contains the sequence to sequence transliteration code
  + Question Codes:
    + question2.py
    + question4_6.py : Contains question_4 and question_6 solution
    + question5.py  


### Set up
+ Install wandb and set up login
+ Install following pacakges
```bash
pip install numpy, matplotlib, PyYAML, sklearn
```
+ Install font for Hindi for matplotlib
```bash
!apt-get install -y fonts-lohit-deva
!fc-list :lang=hi family
```

### Run question 2
```bash
python question2.py
```

### Run question 4 and question 6
```bash
python question4_6.py
```

### Run question 5
```bash
python question5.py
```
