# Inference of Cell Lineage Tree from scRNA-Seq Data
## Project Description

This project aims to develop a pipeline that reconstructs the phylogenetic tree of cells from single-cell transcriptomic data.
The inferred tree takes the form of a (strictly) binary cell lineage tree, in which each node represents a cell and mutations are attached to edges.


## Requirements to Run the Project

The project in written in Python. To be able to run it, one needs the following packages (in alphabetical order):
- graphviz
- numpy
- pandas
- scikit-learn
- scipy


## How to Use the Data Generator

There is a data generator class that uses the same model to generate random data.
Simulated data from the data generator can be used to test the tree inference algorithm.
To create a `DataGenerator` object, we need to specify the number of cells and loci.
```
from tree_inference.data_generator import DataGenerator
dg = DataGenerator(50, 10)
```
Here, `50` is the number of loci and `10` is the number of cells.
We can provide a (cell lineage) tree object to the DataGenerator, or we can tell it to generate a random tree, using the `random_tree` function.
```
dg.random_tree()
```
The tree is stored in `dg.tree` and can be converted to graphviz objects or parent vectors, just as described above.
Note that the tree also contains all mutation locations.
Next, we need to provide to it the real genotypes before and after the mutation at each locus.
Again, this step can be randomized using the `random_mutations` function.
```
dg.random_mutations(mut_prop = 0.5, genotype_freq = [1/3, 1/3, 1/3])
```
When generating random genotypes, we need to specify the proportion of "real" mutations (i.e. proportion of loci that indeed contain a mutation), while the rest are "fake mutations" (for which the two genotypes are the same).
In addition, we need to provide the frequency of the three genotypes in wildtype cells.

Finally, we can get the random data by using the function `generate_reads`.
```
ref, alt = dg.generate_reads()
```


## How to Perform a Tree Inference

It is assumed that the data are in the form of counts of reference and alternative reads for all cells and loci. More specifically, the data should consist of two matrices, say $R$ and $A$, in which $R_{i,j}$ and $A_{i,j}$ are the counts of reference and altertive reads in cell i at locus j, respectively.

To perform tree inference, we first need to identify the mutated loci.
For each mutated locus, since the model assumes that each locus has at most 1 mutation (see the report for details), we also need to infer the genotypes before and after the mutation.
We can do this with the help of the `MutationFilter` object.
```
from tree_inference.mutation_filter import MutationFilter
mf = MutationFilter()
selected, gt1, gt2 = mf.filter_mutations(ref, alt, method = 'threshold', t = 0.5)
```
Here, `ref` and `alt` are matrices containing the reference and alternative read counts.
The function returns three 1D arrays of the same length. Among them, `selected` contains the indices of loci that are considered mutated, while `gt1` and `gt2` contain the genotypes before and after mutation, respectively.

With mutated loci and the corresponding genotypes, we can now calcualte the log-likelihoods of each cell having genotype 1 and 2.
This is obtained by the method `get_llh_mat` of `MutationFilter`.
```
llh_mat_1, llh_mat_2 = mf.get_llh_mat(ref[:,selected], alt[:,selected], gt1, gt2)
```

The inference process is handled by the class `TreeOptimzier`.
We create a `TreeOptimizer` instance and prepare it for inference by providing the log-likelihood matrices obtained above to its `fit` method.
```
from tree_inference.tree_optimizer import TreeOptimizer
optz = TreeOptimizer()
optz.fit(llh_mat_1, llh_mat_2, reversible = True)
```
Here, the argument `reversible` tells the object whether it should also consider the reverse mutation (i.e. from `gt2` to `gt1`).

After fitting, we start the inference and wait for it to finish (which can take some time, especially for large trees).
```
optz.optimize(spaces = ['m', 'c'])
```
Here, the argument `spaces` specifies which tree spaces the algorithm should explore, as well as in which order they should be explored.
Each space is represented by a single character. 
For example, `'m'` stands for the space of mutation trees, and `'c'` stands for the space of cell lineage trees.

Once the inference is finished, the resulted cell lineage tree and mutation tree are stored in the object respectively as `optz.ct` and `optz.mt`.
Both of the trees can be converted to a `graphviz.Digraph` object using the member function `to_graphviz`.
```
optz.ct.to_graphviz()
optz.mt.to_graphviz()
```
When using Jupyter Notebook, a `graphviz.Digraph` object can be displayed directly in the notebook.
Otherwise, it can be saved with the function `graphviz.Digraph.save`.
Alternatively, a cell lineage tree can also be converted to a parent vector as `optz.ct.parent_vec`, which can then be saved in various forms.
Note that the parent vector doesn't contain information on mutation attachments.
If needed, the mutation attachments can be obtained separately as `optz.ct.attachments`.
Finally, the history of likelihood (i.e. joint likelihood of the tree given the data) during the inference can be found as `optz.likelihood_history`.


## Multiple Tests

The function `test_inference` allows repetitively testing the same combination of `DataGenerator`, `MutationFilter` and `TreeOptimizer` objects.
```
from test_functions import test_inference
test_inference(dg, mf, optz)
```
This function runs the entire pipeline (i.e. data generatioin, mutation filtering and tree optimization) for a defined number of times (10 times by default) and it returns some statistics that help evaluate the performance of the pipeline. The returned statistics are:
- distance to the real tree
- runtime
- adjusted likelihood compared to that of the real tree


## External Links
