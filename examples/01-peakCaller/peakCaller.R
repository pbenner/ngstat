
source("../../rUtility/distributions.R")

## -----------------------------------------------------------------------------

mixture       <- fromJSON(file="peakCaller.json")
nonparametric <- fromJSON(file="../data/Liver-Day12.5-H3K27ac.raw.json")$Parameters

## -----------------------------------------------------------------------------

x <- nonparametric$X

pi <- mixture$Parameters
d1 <- parse.distribution(mixture$Distributions[[1]])
d2 <- parse.distribution(mixture$Distributions[[2]])
d3 <- parse.distribution(mixture$Distributions[[3]])
d4 <- parse.distribution(mixture$Distributions[[4]])
bg <- function(x) pi[1]*d1(x) + pi[2]*d2(x)
fg <- function(x) pi[3]*d3(x) + pi[4]*d4(x)

col=c("darkolivegreen3", "darkolivegreen4", "firebrick3", "firebrick4")

## -----------------------------------------------------------------------------

plot(Y ~ X, nonparametric, type="l", ylab="log density")
points(x, log(pi[1]*d1(x)), col=col[1], lwd=2)
lines (x, log(pi[2]*d2(x)), col=col[2], lwd=2)
lines (x, log(pi[3]*d3(x)), col=col[3], lwd=2)
lines (x, log(pi[4]*d4(x)), col=col[4], lwd=2)
legend("topright", legend=c("Background 1", "Background 2", "Foreground 1", "Foreground 2"), col=col, lwd=2, bty="n", pch=c(1,NaN,NaN,NaN), lty=c(NaN,1,1,1,1))

## -----------------------------------------------------------------------------

plot(Y ~ X, nonparametric, type="l", ylab="log density", xlim=c(0,100))
lines(x, log(bg(x)), col=col[2], lwd=2)
lines(x, log(fg(x)), col=col[4], lwd=2)
legend("topright", legend=c("Background", "Foreground"), col=col[c(2,4)], lwd=2, bty="n")

## -----------------------------------------------------------------------------

plot(Y ~ X, nonparametric, type="l", ylab="log density", xlim=c(0,100))
lines(x, log(bg(x)+fg(x)), col=col[4], lwd=2)
