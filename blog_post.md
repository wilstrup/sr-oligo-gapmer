# Some title

## Introduction

### Transcrpition process
DNA -> RNA -> Protein

### Why and How do ASOs work?
`A protein is being produced and being produced by a gene somewhere and causes something bad to happen. Why do we want to stop it being produced?`

We intervene in the RNA phase to stop the transcription into proteins.

We do this by binding DNA to RNA which is an unnatural structure in the body. So the RNAse destroys the RNA thus preventing the bad protein from being produced.

`picture here`

The challenger here is to take this synthesised strand of DNA (also known as ASOS) and attach it to the target. A single strand of DNA does not last long in the body so we need to protect this DNA. One way of protecting it is with Locked Nucleic Acids.

### What are LNA

LNA is a modified RNA nucleotide that improves stability and reduces degradation caused by enzymes. More specifically, the DNA is flanked by a combination of LNA and DNA bases. In this notebook we will write ASOs as strings of nucleobases with the convention that bases that have been modified to be LNA will be written in uppercase, while unmodified DNA will be written with the base letter in lowercase, like this:

<big> ```TGGCaagcatccTGTA``` </big>


### Gapmers -> the thing above
Gapmers are flanked ASOs. In this case, we have more specifically, LNA-gapmers.

### Toxic side effects

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