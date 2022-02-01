# gperceiver

Work-in-progress package to train and use a [Perceiver](https://arxiv.org/abs/2103.03206) inspired neural network for genomic prediction.
We have a pretraining step based on [Perceiver IO](https://arxiv.org/abs/2107.14795) to learn appropriate embeddings.

Currently this uses a BERT style encoder-decoder pretraining scheme, where 10% of the markers are input as missing values and the decoder predicts these masked tokens.
In the near future I intend to add a contrastive pretraining mode, where the model takes two genotypes, runs them through a twinned/siamese network and is asked to predict how many alleles the two share at a number of randomly selected positions.
What I've found with the BERT style pretraining, the model tends to just learn where other markers with the same value (i.e. 0 (aa), 1 (Aa), 2 (AA)) are.
The contrastive learning should mask this information and force the model to learn relationships between markers and individual genotypes.

The finetuning step involves adding a new trainable query for each target environment to the latent vectors, which then allows those vectors to easily retrieve novel information from the cross-layers, as well mixing with the existing latent vectors and other environmental vectors in the self-attention layers.
Alternatively, we can average-pool the latent vector position-wise (as in the original Perceiver paper), to obtain a final embedded representation of the model, and train a new model on top.
The former has the advantage of explicit information sharing between environments, and it should be complex enough to not require setting all of the weights to be trainable in the finetuning stage, which should prevent over-fitting.

Overall, the idea is that by pre-training on the markers in all of the genotypes (including the test, since we only really use these models once), we can avoid the drop in performance caused by loss of linkage between generations.
It also enables the inclusion of more data from external sources, or exertion of existing knowledge e.g. restricting positional embeddings using graph neural networks.

More details to come.
