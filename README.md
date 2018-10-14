# JointBioER

***
## Dependency package

JointBioER uses the following dependencies:

- [Python 2.7](https://www.python.org/)
- [keras 2.0.3](https://keras.io/)
- [Tensorflow 1.1.0](https://www.tensorflow.org/)
- [numpy 1.12.1](http://www.numpy.org/)

## Content
- JointBioER
	- DDI: the data and source code of our JointBioER on the DDI 2013 corpus
	- CPR: the data and source code of our JointBioER on the CPR corpus

- Pipeline
	- DDI
		- NER_DDI: NER of pipeline on the DDI 2013 corpus
		- RE_DDI: RE of pipeline on the DDI 2013 corpus
	- CPR
		- NER_CPR: NER of pipeline on the CPR corpus
		- RE_DDI: RE of pipeline on the CPR corpus



## Train a pipeline model
To train a pipeline model, you need to train a NER model, and then train a RE model.

for example:
```
cd ./Pipeline/DDI/NER_DDI/src
python bilstm_crf_train.py
cd  ./Pipeline/DDI/RE_DDI/src
python bilstm_re_keras.py
```

## Train a JointBioER model
Train our JointBioER model on the DDI 2013 corpus:

```
cd ./JointBioER/DDI/src
python bilstm_crf_train_ddi.py 
```

Train our JointBioER model on the CPR corpus:

```
cd ./JointBioER/CPR/src
python bilstm_crf_train_cpr.py 
```
