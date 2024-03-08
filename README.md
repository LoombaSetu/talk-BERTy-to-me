# Summary of work
Our goal was to create a story generator that generates coherent sentences conditioned on
certain inputs like genre and title. We approached it from 2 directions: one, use an off the shelf
model like GPT2, and two, build our own RNN-based Seq2Seq architecture. The goal of this
two-fold approach was to have something functional, while also learning the complexities
associated with material learned in the class. In terms of output, right now, using 400 books
from Project Gutenberg, our fine-tuned GPT2 model generates text for genres such as fiction,
history, non-fiction etc. 

# Implementation
Seq2Seq.ipynb details the implementation of the Seq2Seq based architecture <br>
AWS_GPT-2_full_dataset_final.ipynb details the implementation of the fine-tuning architecture
