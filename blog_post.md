# Some title

## Antisense therapy

Since the advent of the human genome project, we have advanced rapidly in sequencing technology. One big advantage to this is that we have been able to identify the genetic roots of several diseases such as cancer, Parkinson's, rheumatoid arthritis, and Alzheimer's. This knowledge has been thoroughly applied to diagnostic use. However, its full potential has not been realised to treating these diseases.

How wonderful would it be to target these specific genetic defects with personalized medicine?

Recently, RNA- and DNA-based drugs have shown great promise of treating diseases at the genetic level. Typically they are chemically engineered oligonucleotides that are complementary to a specific messenger RNA (mRNA). They bind to the mRNA through standard base-pairing which stops the translation of the target protein. These short, synthetic, single-stranded nucleic acids are called antisense oligonucleotides (ASOs).

`insert picture of ASO targeting an mRNA`

Include reference to red paper

## Locked Nucleic Acids (LNA)

If we simply produce a short strand of DNA or RNA complementary to the specific mRNA we run into some limitations:

* it will be unstable, i.e. likely to degrade;
* it will likely bind to other RNA sequences than the targeted one, i.e. low specificity;

and these cause a lot of side effects. Therefore the oligonucleotides need modifications to overcome these limitations. One example are locked nucleic acids (LNA). It is a modification of the sugar ring that make up the nucleotide.

Include Jesper Wengle review paper on LNA as a novel class....

## LNA-Gapmers

How to make use of LNA, then? The synthesized oligonucleotide above will be flanked by LNA molecules, which not only will protect it from enzymes, but also help it attach to the targeted mRNA sequence. These are called LNA-gapmers and the gap refers to the unmodified oligonucleotide between the LNA flanks.

The ASOs we will be discussing in this blogpost are LNA-gapmers. Here they will be represented as strings of nucleobases with the convention that bases that have been modified to be LNA will be written in uppercase, while unmodified nucleotides will be written with the base letter in lowercase. Below is an example with a DNA oligonucleotide:

`pciure of oligonucleotide with LNA flanks`

<big> ```TGGCaagcatccTGTA``` </big>

## Toxic side effects

Some LNA ASOs are very effective at reducing expression of targeted proteins. However there is evidence that LNA ASOs can cause liver damage (hepatotoxicity).

[ref] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1802611/.
[ref] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4797265/

While such drugs would not be approved for clinical trials, a clear understanding of the mechanism leading to toxicity is necessary to improve the development of safe ASO drugs.

Preclinical studies have shown that levels of liver toxicity were independent of the gene inhibition caused by LNA ASOs [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1802611/]. Moreover the replacement of a single nucleotide in the LNA gapmer can have significant effects on its toxicity [https://www.liebertpub.com/doi/abs/10.1089/nat.2012.0366], meaning that LNA ASO design could be a major contributer to toxicity.

People have used machine learning methods to understand the relation between LNA ASO sequence structures and toxicity [ref]. In this blogpost we aim at contributing to these efforts by applying a novel method using symbolic regression: the QLattice.

The QLattice is a symbolic regression algorithm designed to find the simplest mathematical relationship that will explain observations. [ref]

In summary, the crux of this blogpost will be to try to answer the following question:

**Does LNA design contribute to toxicity and if so how?**

## Data

In our analysis, we used the data from [Natalia Papargyri et al](https://doi.org/10.1016/j.omtn.2019.12.011). The data set contains two sets of iso-sequential LNA-modified gapmers, where they systematically varied the number and positions of LNA modifications in the flanks.

Specifically, there are 768 different LNA gapmers, where 386 of them target a region we will call **region A**, and 386 of them target a region we will call **region B**. This means the "6-gap" is the same within each region and the only variance is in the LNA flanks. Four of the ASOs target neither of the regions and were included in the original study as inactive controls. We will not use those four ASOs in this blogpost.

> For the curious: Both region A and B are located on the hypoxia-inducible factor 1-alpha (HIF1A) mRNA. "The HIF1A protein regulates cellular responses to hypoxia, and elevated expression of HIF1A has been associated with poor prognosis for many types of cancer."

In the aforementioned study toxicity is measured by caspase activation. Caspase is a family of enzymes that play an essential role in programmed cell death.

Below are the first five entries of the data set.

`show data.head()`

- **target**: A or B, indicates which region in HIF1A is targeted.
- **sequence**: The sequence of the ASO. lowercase means DNA, uppercase means LNA
- **design**: The gapmer design. Each character is either L or D for LNA or DNA.
- **cas_avg**: The average caspase activation across several measurements. Low caspase means that the ASO is safe, high caspase means that the ASO is toxic. This is a percentage of the usual baseline levels of caspase activity in a healthy cell.
- **kd_avg**: Average knockdown. How potent the ASO is in reducing the targeted mRNA. It is expressed as a percentage of the mRNA remaining, so a low value means a potent drug, while a value close to 100 means the mRNA wasn't knocked down by the ASO.

## Strategy

First we will use the QLattice to find a mathematical expression that will serve as a hypothesis for the relation between LNA ASO design and toxicity solely on region A. Then we will scrutinize this hypothesis by testing whether it generalises to region B.

Previous work has shown that a reasonable threshold for caspase activation is 300%. [(Deickmann, 2018)](https://doi.org/10.1016/j.omtn.2017.11.004).
We will use this as the cutoff value for training the QLattice classifier model: below this value the drug is seen as having low/mid levels of toxicity (negative class) while above this threshold the drug is seen as very toxic (positive class).

It should be noted that we will not be optimizing for potency (knockdown).

`Histogram of CA on each region with threshold`

Since we know that the ASOs within their respective targeted regions differ only in the where the LNA modifications are, we will be generating a model that explains caspase activation from the LNA/DNA configuration only. For example each ASO within region A consists of or is a subset of the following sequence

<big> ```tggcaagcatcctgta``` </big>

One example of an LNA modified ASO in the data set is this sequence.

<big> ```TggcAagcatccTgTA``` </big>

When it comes to the feature engineering we will count only the amount of  upper case (LNA) and lower case (DNA) bases in the flanks.

### Feature engineering

We will engineer four features that capture some of the LNA ASOs design.

- **lna_5p**: The number of LNA nucleobases in the *left* flank (5'). Embedded DNA bases are not counted
- **lna_3p**: The number of LNA nucleobases in the *right* flank (3'). Embedded DNA bases are not counted
- **lna_count**: The number of LNA nucleobases across the ASO
- **dna_count**: The number of DNA nucleobases across the ASO

Note that there is some redundancy here: $lna\_count = lna\_5p + lna\_3p$. We include it nonetheless to allow the QLattice to choose whether to distinguish between the 5' and the 3' end or not.

Here is the final dataset to be fed to the QLattice. Observe this is only on ASOs from region A.

`dfA.head of final dataset`

## What is a QLattice?

We want to find a mathematical hypothesis that models the toxicity of ASOs with the features we've engineered above. We do this using the QLattice.

The QLattice is a quantum simulator that explores the space of all mathematical expressions, including parameters, for the expressions that best model the relationship between the output (toxicity) and the input (ASO design characteristics). The result of the search is a list of hypotheses sorted by how well they match observations.

In this blogpost we use the QLattice to generate classification models. Mathematically, this means that the QLattice will wrap each expression in a logistic function. This allows the output to be interpreted as a probability. In other words: if $X$ is an input vector, $f(X)$ is the mathematical equation, and $Y$ is the event we want to predict, then the QLattice will search for functions $f$ such that the predictive power of:
$$\widehat{Y} = \frac{1}{1+e^{-f(X)}}$$
is maximised.

In our case, Y is the probability of an ASO being above the 300% toxicity cutoff value, and $f$ is *any* function of our input features:

$$\widehat{P}(too\_toxic) = \frac{1}{1+e^{-f(lna\_5p, lna\_3p, dna\_count, lna\_count, aso\_len)}}$$

> Note that in order to connect to and use a QLattice you need get one from Abzu here: link to website.

## Finding hypotheses

Below is a code snippet of how to search for hypotheses with the QLattice.

```python
hellow world
```

## Results

Here is the best performing graph (hypothesis) according to AIC:

`picture of graph`

Here is the graph interpreted as a mathematical expression:

`pciture of sympify`

`metric plots, ROC curves, probability plots`

`partial 2dplots`

### Testing on region B

`metric plots, ROC curves, probability plots`

`partial 2dplots`

## Conclusion

We began our journey settled on tackling the question:

**Does LNA design contribute to toxicity and if so how?**

Our hypothesis indicates that LNA design **does** contribute to toxicity. More specifically, it says that high toxicity is associated with high LNA count ($\gtrsim$ 3) on both flanks. Furthermore, the good performance on region B added validity to the hypothesis.

To provide further tests to this hypothesis it would be interesting to perform experiments with LNA ASOs whose sequences are different from the ones in regions A and B.

Note that in our feature engineering process we didn't discriminate between positions of the LNA modifications in the flanks, we only counted them. This could be a cause to the misclassifications as some positions might contribute more than others. For further study we could investigate whether the position of the LNA modification in the flanks contributes to toxicity.

----

# Important phrases/keywords

Overarching theme: able to find a general model based on certain ASO features to predict their toxicity using simple mathematical equations.

Scientific method: Formulate question (Does LNA design contribute to CA and if so how?), hypothesise (mathematical formulas that we get out), prediction, testing (test on new data (region B)), analysis (interpretation)