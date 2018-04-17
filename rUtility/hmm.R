
library(grid)
library(rjson)

## -----------------------------------------------------------------------------

normalize.matrix <- function(x, bycol=FALSE) {
    if (bycol) {
        a <- apply(t(x), 1, min)
        b <- apply(t(x), 1, max)
        t((t(x)-a)/(b-a))
    } else {
        (x-min(x))/(max(x)-min(x))
    }
}

parse.matrix <- function(json, n, m) {
    matrix(json, n, m, byrow=TRUE)
}

## -----------------------------------------------------------------------------

parse.hmm.tree <- function(json) {
    if (is.null(json$Tree)) {
        return()
    }
    n <- length(json$P0)
    r <- data.frame(L0 = rep(0, n))
    c <- 1
    f <- function(tree, i) {
        if (is.null(tree$Children)) {
            s1 <- tree$States[1]+1
            s2 <- tree$States[2]
            r[s1:s2,i+1] <<- c
            c <<- c+1
        } else {
            r[,sprintf("L%d",i+1)] <<- i
            for (node in tree$Children) {
                f(node, i+1)
            }
        }
    }
    f(json$Tree, 0)
    r
}

## -----------------------------------------------------------------------------

parse.hmm <- function(json, import.edist=TRUE, summarize.edist=TRUE) {
    edist <- NULL
    if (import.edist) {
        if (summarize.edist) {
            for (distribution in json$Distributions) {
                r <- parse.distribution.summarized(distribution)
                if (class(r) == "matrix") {
                    edist[[length(edist)+1]] <- r
                } else {
                    edist <- rbind(edist, r, deparse.level=0)
                }
            }
        }
        else {
            for (distribution in json$Distributions) {
                r <- parse.distribution(distribution)
                if (class(r) == "matrix") {
                    edist[[length(edist)+1]] <- r
                } else {
                    edist <- rbind(edist, r, deparse.level=0)
                }
            }
        }
    }
    list(
        Pi = json$Parameters$Pi,
        Tr = parse.matrix(json$Parameters$Tr, json$Parameters$N, json$Parameters$N),
        EDist = edist,
        Tree  = parse.hmm.tree(json),
        StateMap = json$Parameters$StateMap)
}

## -----------------------------------------------------------------------------

import.hmm <- function(filename, ...) {
    parse.hmm(fromJSON(file=filename), ...)
}
