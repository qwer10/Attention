## Descriptioin
Using an attention model to build a Neural Machine Translation (NMT) model to translate human readable dates ("25th of June, 2009") into machine readable dates ("2009-06-25").


### Model
the architecture as follows:
![image](https://github.com/qwer10/Attention/blob/master/images/attn_mechanism.png)
![image](https://github.com/qwer10/Attention/blob/master/images/attn_model.png)


## Requirements
1. TensorFlow 
2. python 3 or later
3. please import Keras, faker, tqdm, numpy

## Usage
```
python Attention.py
```
## Resualts
source: 3 May 1979 

output: 1979-05-03

source: 5 April 09

output: 2009-05-05

source: 21th of August 2016

output: 2016-08-21

source: Tue 10 Jul 2007

output: 2007-07-10

source: Saturday May 9 2018

output: 2018-05-09

source: March 3 2001

output: 2001-03-03

source: March 3rd 2001

output: 2001-03-03

source: 1 March 2001

output: 2001-03-01

