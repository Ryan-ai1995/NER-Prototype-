# Named Entity Recognition - Prototype

Hello!

These scripts form a demonstration of a Named Entity Recognition system that can be applied to books, or any form of text that is in the .txt format.

A list of required packages is given below and a full list is provided as a .txt file in this repo.

1) NLTK
2) Pandas
3) Numpy
4) Pytorch
5) Transformers - Hugging Face
6) FastAPI
7) Uvicorn

Please also note that the first time you run these scripts, a deep learning model from Hugging Face will be downloaded into the working directory - Details on this model can be found here:
https://huggingface.co/dslim/bert-base-NER

I have created a conda environment for this project to keep everything in one place and track each package and its version that is being used.

Three weblinks to different books (.txt files) were given as part of this project. These are given below as I have tested the code on all of them and the NER results for each book can also be found in this repository. I have also included a photo of the output on my local host server, through FastAPI.

https://www.gutenberg.org/cache/epub/2447/pg2447.txt (This is Eminent Victorians by Lytton Strachey and the runtime for this file is roughly 1600 seconds on a Macbook M1 Pro)

https://www.gutenberg.org/cache/epub/64317/pg64317.txt (This is The Great Gatsby by F. Scott Fitzgerald and the runtime for this file is roughly 250 seconds on a Macbook M1 Pro as it is shorter)

https://www.gutenberg.org/cache/epub/345/pg345.txt (This is Dracula by Bram Stoker and the runtime for this file is roughly 2100 seconds on a Macbook M1 Pro)

These are quite lengthy runtimes and so this should be treated as a demonstration project only. In a live business project, requirements would need to be discussed within the team and with clients to decide upon the best balance between performance and processing time. There are other packages such as Spacy and NLTK that also can perform Named Entity Recognition, which I have looked into and have a smaller footprint. Performance comparisons between each of these could be conducted.

Also note that this Hugging Face model is also not perfect. From my observations, it can mistake person entities for location entities, with one example being the model predicting that 'Dracula' was a 
location entity. It also has a token limit of 512 tokens, which means that the text must be split up into partitions and processed one after the other. This adds complexity to the code. Additionally, tokenisation of the input .txt file can greatly affect the results. The tokenizer that comes alongside the deep learning model often split the text in such a way
that many of the resulting tokens were not words. 

It is for this reason that I used the NLTK tokenizer to tokenize the text (it seems to be cleaner) and then removed any tokens that I did not consider to be words such as roman numerals, non-alphanumeric
characters etc. I then created a list of 'words' for each chunk of text and referred to this when deciding upon the search radius for location entities.

Further work would involve further formatting of the code to make it more maintainable, looking for ways to optimise the runtime and exploring other Named Entity Recognition Models.
