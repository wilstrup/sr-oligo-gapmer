# Some title

## Antisense therapy

Since the advent of the human genome project, we have advanced rapidly in sequencing technology. One big advantage to this is that we have been able to identify the genetic roots of several diseases such as cancer, Parkinson's, rheumatoid arthritis, and Alzheimer's. This knowledge has been thoroughly applied to diagnostic use. However, its full potential has not been realised to treating these diseases.

How wonderful would it be to target these specific genetic defects with personalized medicine?

Recently, RNA- and DNA-based drugs have shown great promise of treating diseases at the genetic level. Typically they are chemically engineered oligonucleotides that are complementary to a specific messenger RNA (mRNA). They bind to the mRNA through standard base-pairing which stops the translation of the target protein. These short, synthetic, single-stranded nucleic acids are called antisense oligonucleotides (ASOs).

`insert picture of ASO targeting an mRNA`

Include reference to red paper

## Locked Nucleic Acids (LNA)

If we simply produce a short strand of DNA or RNA complementary to the specific mRNA we run into some limitations:

* It will be unstable, i.e. likely to degrade.
* It will likely bind to other RNA sequences than the targeted one.

and these cause a lot of side effects. Therefore the oligonucleotides need modifications to overcome these limitations. One example are locked nucleic acids (LNA). Specifically LNA is a modification of the sugar ring that make up the nucleotide.

Include Jesper Wengle review paper on LNA as a novel class....

## LNA-Gapmers

How to make use of LNA, then? The synthesized oligonucleotide above will be flanked by LNA molecules, which not only will protect it from enzymes, but also help it attach to the targeted mRNA sequence. These are called LNA-gapmers and the gap refers to the unmodified oligonucleotide between the LNA flanks.

The ASOs we will be discussing in this blogpost are LNA-gapmers. Here they will be represented as strings of nucleobases with the convention that bases that have been modified to be LNA will be written in uppercase, while unmodified nucleotides will be written with the base letter in lowercase. Below is an example with a DNA oligonucleotide:

<big> ```TGGCaagcatccTGTA``` </big>

## Toxic side effects

Some LNA ASOs are very effective at reducing expression of targeted proteins. However there is evidence that LNA ASOs can cause liver damage (hepatotoxicity).

[ref] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1802611/.
[ref] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4797265/

While such drugs would not be approved for clinical trials, a clear understanding of the mechanism leading to toxicity is necessary to improve the development of safe ASOs drugs

Preclinical studies have shown that levels of liver toxicity were independent of the gene inhibition caused by LNA ASOs [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1802611/]. Moreover the replacement of a single nucleotide in the LNA gapmer can have significant effects on its toxicity [https://www.liebertpub.com/doi/abs/10.1089/nat.2012.0366], meaning that LNA ASO design could be a major contributer to toxicity.

----

However, it is not obvious how the LNA ASO sequence composition causes such liability, as their degrees of toxicity can vary widely.



People have used machine learning methods to understand the relation between LNA ASO sequence structures, toxicity and potency


The general challenges of using ASOS. The side effect of using strong bindings such as LNA is that the can interact with other things they shouldn't be.

What happens to make it toxic? Does it stop the production of useful proteins?

In practice many ASOs have a significant cytotoxic potential (their presence in the cell causes the cell to die). Often even small variations in the design of an ASO can change it from non-toxic to highly toxic. Even for ASOs that have the exact same sequence of nucleobases, the toxicity can vary with other modifications to the molecule.

### The pattern of toxicity and motivation for QL

A lot of research is done on what causes the toxicity of ASOs [ref].

In this blogpost we investigate the problem in a simple manner: we find a mathematical expression that generalises over two different ASOs.

Many studies have applied traditional machine learning methods such as random forest and gradient boosting to predict ASO toxicity [ref]. However, they often did not generalise to new experiments [ref].

We use the QLattice to find the mathematical expression.

The QLattice is a symbolic regression algorithm designed to find the simplest mathematical relationship that will explain observations.

**The question: Does LNA design contribute to CA and if so how?**


# Data

The data set used in this notebook comes from a [2019 research paper by Natalia Papargyri et al](https://doi.org/10.1016/j.omtn.2019.12.011) from Roche Pharma Research. The data set contains the design of 768 different gapmers. The gapmers target one of two regions on the mRNA. 386 of them target a region we will call **region A**, and 386 of them target a region we will call **region B**. This means the "6-gap" is the same within each region and the only variance is in the LNA flanks. Four of the ASOs target neither of the regions and were included in the original study as inactive controls. We will not use those four ASOs in this blogpost.

"we analyzed two sets of iso-sequential locked nucleic acid (LNA)-modified gapmers, where we systematically varied the number and positions of LNA modifications in the flanks"

> For the curious: Both region A and B are located on the hypoxia-inducible factor 1-alpha (HIF1A) mRNA. The HIF1A protein regulates cellular responses to hypoxia, and elevated expression of HIF1A has been associated with poor prognosis for many types of cancer.

### Measuring toxicity
Check what the paper says about how they measure it.

`show data.head()`

- **target**: A or B, indicates which region in HIF1A is targeted.
- **sequence**: The sequence of the ASO. lowercase means DNA, uppercase means LNA
- **design**: The gapmer design. Each char is either L or D for LNA or DNA.
- **cas_avg**: The average caspase activation across several measurements. Low caspase means that the ASO is safe, high caspase means that the ASO is toxic
- **kd_avg**: Average knockdown. How potent the ASO is in reducing the targeted mRNA. It is expressed as a percentage of the mRNA remaining, so a low value means a potent drug, while a value close to 100 means the mRNA wasn't knocked down by the ASO.

# Method:

The objective is to find a mathematical model of toxicity that generalises over both region A and region B.

So that's the current strategy: train on Region A and test on Region B for CA single target optimization (because we just want to optimize for toxicity). In other words, we don't care about molecule stability, only toxicity.

`Histogram of CA on each region with threshold`

What is the problem with regression? Numbers can vary from study to study. We want to paint a general picture of CA in ASOs.

Previous work has shown that reasonable threshold for caspase activation is 300%. [(Deickmann, 2018)](https://doi.org/10.1016/j.omtn.2017.11.004). We will use this as the cutoff value for training the QLattice classifier model


### Feature engineering, 5', 3' etc

We will engineer four features based on the design that capture some of the ASOs attributes.

- **lna_5p**: The number of LNA nucleobases in the *left* flank. Embedded DNA bases are not counted
- **lna_3p**: The number of LNA nucleobases in the *right* flank. Embedded DNA bases are not counted
- **lna_count**: The number of LNA nucleobases across the ASO
- **dna_count**: The number of DNA nucleobases across the ASO

Note that there is some redundancy here: $ lna\_count = lna\_5p + lna\_3p $. We include it nonetheless to allow the QLattice to choose whether to distinguish between the 5' and the 3' end or not.

`df.head of final dataset`

# What is the QLattice etc

We want to find a mathematical expression that models the toxicity of an ASOs with the features we've engineered above. We do this using the QLattice (what are the usual motivations for symbolic regression?).

The QLattice explores the space of all mathematical expressions, including parameters, for the expressions that best model the relationship between the output (toxicity) and the input (ASO design characteristics). The result of the search is a list of expressions sorted by how well they match observations.

In this blogpost we use the QLattice to generate classification models. Mathematically, this means that the QLattice will wrap each expression in a logistic function, which allows the output to be interpreted as a probability. In other words: if $X$ is an input vector, $f(X)$ is the mathematical equation, and $Y$ is the event we want to predict, then the QLattice will search for functions $f$ such that the predictive power of:
$$\widehat{Y} = \frac{1}{1+e^{-f(X)}}$$
is maximised.

In our case, Y is the probability of an ASO being above the 300% toxicity cutoff value, and $f$ is *any* function of our input features:

$$\widehat{P}(too\_toxic) = \frac{1}{1+e^{-f(lna\_5p, lna\_3p, dna\_count, lna\_count, aso\_len)}}$$

> Note that in order to connect to and use a QLattice you need get one from Abzu. QLattices are a limited resource, but we do offer them for free for scientific purposes.

# Do the fitting

We do the fitting on only on ASOs that target region A.
```python
qgraph.fit(dfA)
```

# Results
Nice 2dplots that show the hotspot, mathematical function

Test on Region B

ROC curves as well

# Conclusion

Things to say:
* This generalises to region B
* How does this answer the question: Does LNA design contribute to CA and if so how. Answer is ... yes and here's the hotspot of low CA.

# Further studies
The combination of left and right flank LNAs. The QLattice didn't pick up on the sum but the combination. We don't discrimbinate where the LNA are in the flanks, we only count them. This could be a cause the missclassifications as some positions might contirbute more than others. So a possible next question could be that does the place in the flank matter?

# Important phrases/keywords

Overarching theme: able to find a general model based on certain ASO features to predict their toxicity using simple mathematical equations.

Scientific method: Formulate question (Does LNA design contribute to CA and if so how?), hypothesise (mathematical formulas that we get out), prediction, testing (test on new data (region B)), analysis (interpretation)

is it obvious that LNA design contribute to CA or is it not obvious because it's case dependent? I don't know