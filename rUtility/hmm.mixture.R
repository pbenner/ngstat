library(rjson)

## -----------------------------------------------------------------------------

hmm.reduce.to.mixture <- function(hmm) {
    eigensystem <- eigen(t(hmm$Tr))

    eigenvalues <- eigensystem$values
    eigenvector <- NULL

    for (i in 1:length(eigenvalues)) {
        if (abs(eigenvalues[i] - 1.0) < 1e-12) {
            eigenvector <- Re(eigensystem$vectors[,i])
            break
        }
    }
    if (is.null(eigenvector)) {
        stop("no eigenvalue of one found")
    }
    eigenvector <- eigenvector/sum(eigenvector)

    ## update P0
    hmm$Pi <- eigenvector
    ## update transition matrix
    for (i in 1:length(hmm$Tr)) {
        hmm$Tr[i] <- eigenvector[((i-1) %% length(eigenvector)) + 1]
    }
    hmm
}
