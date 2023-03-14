# Inference of Cell Lineage Tree from scRNA-Seq Data
## Project Description
This project aims to develop a pipeline that reconstructs the phylogenetic tree of cells from single-cell transcriptomic data.
The inferred tree takes the form of a (strictly) binary cell lineage tree, in which each node represents a cell and mutations are attached to edges.


## Requirements to Run the Project
The project in written in Python. To be able to run it, one needs at least the following packages (in alphabetical order):
- graphviz
- numpy
- pandas
- scikit-learn
- scipy

Additionally, one package is required for plotting:
- matplotlib


## How to Perform a Tree Inference
It is assumed that the data are in the form of counts of reference and alternative reads for all cells and loci. More specifically, the data should consist of two matrices, say $R$ and $A$, in which $R_{i,j}$ and $A_{i,j}$ are the counts of reference and altertive reads in cell i at locus j, respectively.

To perform tree inference, we first need to filter the read counts to identify potentially mutated loci.
Then for the mutated loci, since the model assumes that each locus has at most 1 mutation (see the report for details), we need to infer the genotypes before and after this mutation.
Let ref and alt be the two matrices containing reference and alternative read counts, the loci filtering and genotype inference can be achieved using the function `filter_mutations`.
```
from mutation_detection.py import filter_mutations
ref, alt, gt1, gt2 = filter_mutations(ref, alt, method = 'threshold', t = 0.5)
```
Here, `ref` and `alt` are filtered to contain only columns (loci) that are considered mutated, while `gt1` and `gt2` are arrays containing the genotypes before and after mutation, respectively.
Note that for the purpose of tree inference, the genotypes are not necessary, as long as we know the (log-)likelihoods of each cell having genotype 1 and 2.
This can be obtained by the function `likelihood_matrices`.
```
from mutation_detection.py import likelihood_matrices
likelihoods1, likelihoods2 = likelihood_matrices(ref, alt, gt1, gt2)
```

With the data in appropriate form, we can now turn to the inference itself.
The inference process is handled by the class TreeOptimzier.
We should first create an instance of TreeOptimizer and prepare it for the inference by providing the filtered data via its `fit` member function.
```
from tree_inference import TreeOptimizer
optz = TreeOptimizer()
optz.fit(likelihoods1, likelihoods2, reversible = True)
```
Here, the argument `reversible` tells the object whether it should also consider the reverse mutation (i.e. from `gt2` to `gt1`). (It can also be provided as a list of Boolean values for each locus.)

Then, we can start the inference and wait for it to finish (which can take some time, especially for large trees).
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


## How to Use the Data Generator
There is a data generator class that uses the same model to generate random data.
We can use the data generator to test the tree inference algorithm.
To create a `DataGenerator` object, we need to specify the number of cells and loci.
```
from data_generator import DataGenerator
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



## External Links
