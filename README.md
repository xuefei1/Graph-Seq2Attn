# Graph Sequence-to-Attention Network

This repo contains the code for the G-S2A model, which is presented in paper [*Inferring Search Queries from Web Documents via a Graph-Augmented Sequence to Attention Network*](https://sites.ualberta.ca/~dniu/Homepage/Publications_files/www19-353%20%281%29.pdf)


## How to run
Due to privacy concerns, the original dataset cannot be released with the code.
However, by specifying the input data in the following format, one can easily test this model on his/her own data.

Each data instance should be one line in the input data file, the default expected format is:
```
target query|document content with segmented words . With sentence delimiter specified|document segmented keywords
```
Here, each word is segmented by a single space, and there are three columns separated by |. 
The first column is the truth query to generate.
The second column is the document content, note that the sentence delimiter, here is the ".", is also separated with its surrounding words by a single space.
The third column is a list of keywords from the document, also separated by a single space.

By default, there should be a train data file, a dev data file and a test data file, all placed in the ```data/``` folder.

To change data loading options, such as the sentence or word delimiter, modify the ```read_doc_data()``` function in ```data/read_data.py```.

Execute ```run_graph_seq2attn.py``` to train and evaluate the model.

Modify available hyper-parameters in ```params.py```.

## Requires

Python 3.x

PyTorch 0.4+
