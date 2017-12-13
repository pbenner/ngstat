
library(rjson)

# ------------------------------------------------------------------------------

plot.distribution <- function(json, main="", xlab="", ylab="", lwd=1) {
    if (json$Name == "sequence id") {
        mean <- NULL
        q005 <- NULL
        q095 <- NULL
        for (str in json$Distributions) {
            distribution <- fromJSON(str)
            s <- summarize.distribution(distribution)
            mean <- c(mean, s$mean)
            q005 <- c(q005, s$q005)
            q095 <- c(q095, s$q095)
        }
        n <- length(mean)
        data <- data.frame(x=rep(1:n,3), y=c(mean, q005, q095),
                           group=c(rep(1,n),rep(2,n),rep(3,n)),
                           linetype=c(rep("a",n),rep("b",n),rep("b",n)))
        p <- ggplot(data, aes(x=x, y=y, group=group))
        p <- p + geom_line(aes(linetype=linetype), size=lwd, show.legend = FALSE)
        p <- p + theme_bw()
        p <- p + ggtitle(main)
        p <- p + xlab(xlab)
        p <- p + ylab(ylab)
        p
    } else
    if (json$Name == "mixture distribution") {
        p <- list()
        for (i in 1:length(json$Distributions)) {
            distribution <- fromJSON(json$Distributions[i])
            q <- plot.distribution(distribution, main=sprintf("component weight: %f", json$Weights[i]))
            p <- c(p, list(q))
        }
        p
    } else {
        stop("unknown distribution")
    }
}

plot.distribution.file <- function(file) {
    json <- fromJSON(file=file)
    plot.distribution(json)
}
