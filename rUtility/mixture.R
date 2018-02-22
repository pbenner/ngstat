
## -----------------------------------------------------------------------------

parse.mixture <- function(json, summarize.edist=TRUE) {
    edist <- NULL
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
    list(Weights = json$Parameters,
         EDist = edist)
}

## -----------------------------------------------------------------------------

import.mixture <- function(filename, ...) {
    parse.mixture(fromJSON(file=filename), ...)
}
