
## ------------------------------------------------------------------------------

hmm.set.stationary <- function(filename) {
    json <- fromJSON(file=filename)
    tr   <- parse.matrix(json$Transition)

    eigensystem <- eigen(t(tr))

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
    json$P0 <- eigenvector
    json
}
## json <- reduce.to.mixture("input.json")
## write(toJSON(json), file="output.json")
