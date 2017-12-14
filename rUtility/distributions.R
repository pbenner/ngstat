
library(rjson)

## -----------------------------------------------------------------------------

parse.distribution.summarized <- function(json) {
    if (json$Name == "matrix:vector id") {
        s <- NULL
        for (dist in json$Distributions) {
            s <- rbind(s, parse.distribution.summarized(dist))
        }
        return(s)
    }
    if (json$Name == "matrix:mixture distribution") {
        s <- NULL
        for (dist in json$Distributions) {
            s <- cbind(s, parse.distribution.summarized(dist))
        }
        return(s)
    }
    if (json$Name == "vector:scalar id") {
        s <- c()
        for (dist in json$Distributions) {
            s <- c(s, parse.distribution.summarized(dist))
        }
        return(s)
    }
    if (json$Name == "vector:mixture distribution") {
        s <- json$Parameters
        for (dist in json$Distributions) {
            s <- c(s, parse.distribution.summarized(dist))
        }
        return(s)
    }
    if (json$Name == "scalar:categorical distribution") {
        return(json$Parameters[1])
    }
    if (json$Name == "scalar:beta distribution") {
        return(json$Parameters[1]/(json$Parameters[1] + json$Parameters[2]))
    }
    if (json$Name == "scalar:gamma distribution") {
        return(json$Parameters[1]/json$Parameters[2])
    }
    if (json$Name == "scalar:negative binomial distribution") {
        return(json$Parameters[1])
    }
    if (json$Name == "scalar:log-normal distribution") {
        return(json$Parameters[1])
    }
    if (json$Name == "scalar:normal distribution") {
        return(json$Parameters[1])
    }
    stop(sprintf("could not parse: %s", json$Name))
}

parse.distribution <- function(json) {
    if (json$Name == "matrix:vector id") {
        s <- NULL
        for (dist in json$Distributions) {
            s <- rbind(s, parse.distribution(dst))
        }
        return(s)
    }
    if (json$Name == "vector:scalar id") {
        s <- c()
        for (dist in json$Distributions) {
            s <- c(s, parse.distribution(dist))
        }
        return(s)
    }
    if (json$Name == "scalar:categorical distribution") {
        return(function(x) json$Parameters[x])
    }
    if (json$Name == "scalar:beta distribution") {
        return(function(x) dbeta(x, json$Parameters[1], json$Parameters[2]))
    }
    if (json$Name == "scalar:negative binomial r distribution") {
        return(function(x) dnbinom(round(x[1] + json$Parameters[2]), x[2], 1.0-json$Parameters[1]))
    }
    if (json$Name == "scalar:negative binomial distribution") {
        return(function(x) dnbinom(round(x), json$Parameters[1], 1.0-json$Parameters[2]))
    }
    if (json$Name == "scalar:log normal distribution") {
        return(function(x) dlnorm(x, json$Parameters[1], json$Parameters[2]))
    }
}
