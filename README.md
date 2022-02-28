# gperceiver

Work-in-progress package to train and use a [Perceiver](https://arxiv.org/abs/2103.03206) inspired neural network for genomic prediction.
We have a pretraining step based on [Perceiver IO](https://arxiv.org/abs/2107.14795) to learn appropriate embeddings.

The ultimate goal of this tool is to predict phenotypes in multiple environments given a set of observations.
Currently we support regression and classification targets (ordinal targets hopefully to come), but all targets must have the same objective.

The flexibility of this model means that it:
- natively handles missing genotypes.
- easily handles multi-allelic data.
- supports any ploidy level.
- can be pretrained on large sets of genetic data, before finetuning it to the task.

## A brief overview.

### Pretraining

We have two encoder-decoder pretraining schemes.
The first is similar to the original BERT scheme where a proportion of the markers are input as missing values and the decoder predicts these masked tokens.
This is analogous to common imputation methods, and at prediction on the target tasks means that imputation and prediction should happen simultaneously.
The second method focusses on contrasting two genotypes, and predicting the number of shared alleles for a proportion of loci.
This method uses a twinned network, and is considerably slower and more resource intensive to train.
In both methods the loci used as targets for prediction are sampled in inverse proportion to allele frequencies.
Essentially, we calculate the alternate allele proportions (i.e. 1 - the allele proportion) across the population, then when evaluating a genotype to sample we multiply the actual allele proportions for each copy (and over the twinned samples).
This means that less frequent alleles (and combinations thereof) have a higher likelihood of being sampled.
On average after accounting for the actual frequencies of alleles, the high frequency and low frequency combinations of alleles should be sampled roughly equally.


I've found with the BERT style pretraining that the model tends to just learn where other markers with the same value (i.e. 0 (aa), 1 (Aa), 2 (AA)) are.
The contrastive learning should mask this information and force the model to learn relationships between markers and individual genotypes (which is what we really want).
However, the contrastive task is much more difficult and often it won't converge on a good solution.

### Finetuning.

In the finetuning stage, we again have two options.
The first involves adding a new trainable query for each target environment to the latent vectors, which then allows those vectors to easily retrieve novel information from the cross-layers, as well mixing with the existing latent vectors and other environmental vectors in the self-attention layers.
Alternatively, we can average-pool the latent vector position-wise (as in the original Perceiver paper), to obtain a final embedded representation of the model, and train a new model on top.
The former has the advantage of explicit information sharing between environments, and it should be complex enough to not require setting all of the weights to be trainable in the finetuning stage, which should prevent over-fitting.
In practise, the pooling method yields better performance, but the latent method seems to be less prone to overfitting.

Overall, the idea is that by pre-training on the markers in all of the genotypes (including the test, since we only really use these models once), we can avoid the drop in performance caused by loss of linkage between generations.
It also enables the inclusion of more data from external sources, or exertion of existing knowledge e.g. restricting positional embeddings using graph neural networks.

## Inputs

The inputs we expect are a bit specialised.
Ideally the markers would be provided as a VCF (or similar), but my main use-case data is a table so this is what I'm working with.

### Chromosomes and marker positions

We expect a file detailing the linkage groups/chromosomes and positions of each of the markers.
The file should be a TSV file with the following structure:

```
chr	marker	pos
chr1	marker1	1
chr1	marker2	2.5
chr2	marker2	3
```

Each marker should have a unique name in the marker column.
The position can be a float, so will support either physical or genetic positions (not distances from the previous allele, use a cumulative sum).

### Markers

The markers should be in a tsv file with a "name" column first (indicating your genotype), and a column for each of the markers.
The marker names should match those in the chroms file.

```
name	marker1	marker2	marker3
s1	0	3	5
s2	3	3	5
```

The alleles are encoded as an index to a set of loci.
Essentially in a diploid genome with biallelic markers, we map like so:

```
0 -> [0, 0]
1 -> [0, 1]
2 -> [0, 2]
3 -> [1, 1]
4 -> [1, 2]
5 -> [2, 2]
```

Each set of integers then represents the allele for each of the chromosomes that you have (here 2 for diploid, a hexaploid organism would map to a 6 element set of alleles).
The model does not consider phasing, so the order of the set does not matter (functionall `[1, 2] == [2, 1]`).
We reserve `0` to mean missing data.
So `[0, 0]` means both alleles are missing in a diploid organism.
Other than that the numbers in the mapping represent as many alleles as you have.
The allele number are then used as keys to an embedding lookup in the model, and summed to get the final locus representation.
A mapping for a specific ploidy and number of alleles can be generated using the `gperceiver.preprocessing.gen_allele_decoder` function.
Alternatively, generating every unique and sorted combinatoral ploidy lengthed set of k alleles will work.

I found that files with this information fully provided (e.g. as `A/A`) were just too big to be practical.
An integer encoding is my solution until we decide to support some kind of binary format.

### Phenotypes

The phenotypes also should be a tab separated values file, with the following structure.

```
name	response	block	cv
s1	0.5	0	0
s1	0.1	1	1
s2	-0.5	1	0
s2	-0.25	2	1
```

The `name` should match the name in the markers file for them to be associated.
The `response` can be any float or binary response that you want to predict.
It's best to scale and center continuous response variables, this makes it easier to specify sensible learning rates and avoids the model having to spend a few epochs at the start fitting a (potentially large) intercept (or failing to learn at all).
Having said that i've come across situations before where using unscaled data did improve performance, so your milage may vary.
The `block` is a numerical index representing different environments or years that you want to predict for.
The order isn't important, as long as the same integer always maps to the same blocking factor (start at 0).
The `cv` column indicates whether to consider something as validation data or not.
When running the finetuning step you can specify `--cv 1` to use anything with 1 in this column as validation data.
If you don't want to use a validation set, you can exclude this column.
Other columns in this table are ignored, so you can put other factors or normal names for your `block`s without affecting the results.


### Scripts

We currently have two main scripts.

`gperceiver_pretrain` pretrains the genetic data.
`gperceiver_finetune` fits the pretrained network to a new task (or optionally can train the model from scratch).

I'll add some documentation on how to customise things soon, but for now you can find the available options for the scripts using `--help`.
You can specify the model hyper-parameters by providing a JSON file to the `--params` option.
An example JSON file with the defaults is available in `./test_data/params.json`.
Some of these hyperparameters are also specifyable as command-line arguments, in which case the command line versions are preferred when given.
