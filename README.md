# Inference of Cell Lineage Tree from scRNA-Seq Data
## Project Description

This project aims to develop a pipeline that reconstructs the phylogenetic tree of cells from single-cell transcriptomic data.
The inferred tree takes the form of a (strictly) binary cell lineage tree, in which each node represents a cell and mutations are attached to edges.


## Requirements to Run the Project

This project is written entirely in Python. To be able to run it, one needs the following packages (in alphabetical order):
- graphviz
- matplotlib
- numpy
- pandas
- scikit-learn
- scipy



## Running the demo notebook

In this notebook, we perform a simulation study to test the performance of SCITE_RNA to infer a true tree from read counts.
Specifically, we generate a true tree, convert it to read counts and then analyze our ability to infer the true tree from the read counts.

The workflow contains the following parts:

1. Simulate a random cell lineage tree, representing the true tree.

2. Generate random read counts from the true tree. 

3. Using the reads, call mutations and subsequently filter these by selecting loci considered to be mutated.

4. In these loci, calculate the likelihood for each cell to be mutated or not mutated

5. Perform tree inference. The output is the inferred tree.

6. Calculate the distance between the true tree and the inferred tree




## Compare SCITE_RNA with Dendro


We compared the performance to infer a true tree from read counts between SCITE_RNA and [Dendro](https://doi.org/10.1186/s13059-019-1922-x). 

![PDF](https://github.com/Tigecycline/lab_rotation/blob/new_ideas/dendro_comparison/figures/distances_hist.pdf)

























