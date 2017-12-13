
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

parse.edist.summarized <- function(json) {
    if (json$Name == "multi-track distribution adapter") {
        s <- NULL
        for (dstr in json$Distributions) {
            s <- rbind(s, parse.edist.summarized(fromJSON(dstr)))
        }
        return(s)
    }
    if (json$Name == "sequence id") {
        s <- c()
        for (dstr in json$Distributions) {
            s <- c(s, parse.edist.summarized(fromJSON(dstr)))
        }
        return(s)
    }
    if (json$Name == "mixture distribution") {
        s <- c(json$Weights)
        for (dstr in json$Distributions) {
            s <- c(s, parse.edist.summarized(fromJSON(dstr)))
        }
        return(s)
    }
    if (json$Name == "categorical distribution") {
        return(json$Theta)
    }
    if (json$Name == "beta distribution") {
        return(json$Alpha/(json$Alpha + json$Beta))
    }
    if (json$Name == "gamma distribution") {
        return(json$Alpha/json$Beta)
    }
    if (json$Name == "negative binomial r distribution") {
        return(json$Parameters[1])
    }
    if (json$Name == "negative binomial distribution") {
        return(json$P)
    }
    if (json$Name == "log normal distribution") {
        return(json$Parameters[1])
    }
    if (json$Name == "sv normal distribution") {
        return(json$Parameters[1])
    }
}

parse.edist <- function(json) {
    if (json$Name == "multi-track distribution adapter") {
        s <- NULL
        for (dstr in json$Distributions) {
            s <- rbind(s, parse.edist(fromJSON(dstr)))
        }
        return(s)
    }
    if (json$Name == "sequence id") {
        s <- c()
        for (dstr in json$Distributions) {
            s <- c(s, parse.edist(fromJSON(dstr)))
        }
        return(s)
    }
    if (json$Name == "categorical distribution") {
        return(function(x) json$Theta[x])
    }
    if (json$Name == "beta distribution") {
        return(function(x) dbeta(x, json$Alpha, json$Beta))
    }
    if (json$Name == "negative binomial r distribution") {
        return(function(x) dnbinom(round(x[1] + json$Parameters[2]), x[2], 1.0-json$Parameters[1]))
    }
    if (json$Name == "negative binomial distribution") {
        return(function(x) dnbinom(round(x), json$R, 1.0-json$P))
    }
    if (json$Name == "log normal distribution") {
        return(function(x) dlnorm(x, json$Parameters[1], json$Parameters[2]))
    }
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
            r <- parse.edist.summarized(fromJSON(estr))
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
