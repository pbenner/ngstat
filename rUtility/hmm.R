
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

parse.matrix <- function(json) {
    matrix(json$Values, json$Rows, json$Cols, byrow=TRUE)
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

parse.hmm <- function(json, summarize.edist=TRUE) {
    edist <- NULL
    if (summarize.edist) {
        for (estr in json$Edist) {
            r <- parse.distribution.summarized(fromJSON(estr))
            if (class(r) == "matrix") {
                edist[[length(edist)+1]] <- r
            } else {
                edist <- rbind(edist, r, deparse.level=0)
            }
        }
    }
    else {
        for (estr in json$Edist) {
            r <- parse.edist(fromJSON(estr))
            if (class(r) == "matrix") {
                edist[[length(edist)+1]] <- r
            } else {
                edist <- rbind(edist, r, deparse.level=0)
            }
        }
    }
    list(
        Pi = json$P0,
        Tr = parse.matrix(json$Transition),
        EDist = edist,
        Tree  = parse.hmm.tree(json),
        StateMap = json$StateMap)
}

## -----------------------------------------------------------------------------

import.hmm <- function(filename, ...) {
    parse.hmm(fromJSON(file=filename), ...)
}
