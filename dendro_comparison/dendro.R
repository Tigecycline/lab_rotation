library(DENDRO)


### Demo
# Note: demo$Info and demo$label can be left blank, but demo$Z cannot
#data("DENDRO_demo")
#str(demo)

#demo_qc = FilterCellMutation(demo$X, demo$N, demo$Z, demo$Info, demo$label, cut.off.VAF = 0.05, cut.off.sd = 5)
#str(demo_qc)

#demo_qc$dist = DENDRO.dist(demo_qc$X,demo_qc$N,demo_qc$Z,show.progress=FALSE)
#demo_qc$cluster = DENDRO.cluster(demo_qc$dist,label=demo_qc$label,type='fan')


### functions to test comparison data
read.matrix <- function(path){
  mat <- as.matrix(read.table(path, header=FALSE, sep=" "))
  dimnames(mat) <- NULL
  return(mat)
}

merge.to.parent <- function(merge.mat){
  n.leaves <- nrow(merge.mat) + 1
  result <- rep(-1, 2*n.leaves - 1)
  
  for (i in 1:nrow(merge.mat)){
    for (child in merge.mat[i,]){
      parent.id <- n.leaves + i - 1
      if (child < 0) result[-child] <- parent.id
      else result[n.leaves + child] <- parent.id
    }
  }
  
  return(result)
}

generate.parent.vec <- function(path=NA, n.tests=10){
  if (!is.na(path)) setwd(path)
  pb <- txtProgressBar(min=0, max=n.tests, initial=0, style=3)
  for (i in 0:(n.tests-1)){
    # prepare data needed for tree reconstruction
    ref <- read.matrix(sprintf("ref_%d.txt", i))
    ref <- t(ref)
    alt <- read.matrix(sprintf("alt_%d.txt", i))
    alt <- t(alt)
    mut.indicator <- read.matrix(sprintf("mut_indicator_%d.txt", i))
    mut.indicator <- t(mut.indicator)
    coverage <- ref + alt
    
    # apply DENDRO
    dist <- DENDRO.dist(alt, coverage, mut.indicator, show.progress=FALSE)
    cluster <- DENDRO.cluster(dist, plot=FALSE, type="fan")
    
    # save resulted tree as parent vector
    parent.vec <- merge.to.parent(cluster$merge)
    write.table(parent.vec, sprintf("dendro_parent_vec_%d.txt", i), row.names=FALSE, col.names=FALSE)
    
    setTxtProgressBar(pb, i+1)
  }
  close(pb)
}

#generate.parent.vec("C:\\Users\\bingg\\Desktop\\lab ratation\\comparison_data\\", 100)
generate.parent.vec("D:\\comparison_data\\", 100)
