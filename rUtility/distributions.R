library(rjson)
# ------------------------------------------------------------------------------

parse.matrix <- function(json) {
    matrix(json$Values, json$Rows, json$Cols, byrow=TRUE)
}

summarize.distribution <- function(json) {
    if (json$Name == "normal distribution") {
        mean <- as.numeric(json$Mu$Values)
        variance <- diag(parse.matrix(json$Sigma))
        q005 <- mean - 2*sqrt(variance)
        q095 <- mean + 2*sqrt(variance)
    } else
    if (json$Name == "gamma distribution") {
        alpha <- as.numeric(json$Alpha)
        beta  <- as.numeric(json$Beta)
        mean  <- alpha/beta
        variance <- alpha/(beta*beta)
        q005  <- qgamma(0.05, shape=alpha, rate=beta)
        q095  <- qgamma(0.95, shape=alpha, rate=beta)
    } else {
        stop("invalid distribution")
    }
    list(mean=mean, variance=variance, q005=q005, q095=q095)
}

summarize.distribution.file <- function(file) {
    json <- fromJSON(file=file)
    summarize.distribution(json)
}
