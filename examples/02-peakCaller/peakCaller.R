
source("~/Source/ngstat/rUtility/distributions.R")

## treatment
## -----------------------------------------------------------------------------

mixture       <- fromJSON(file="peakCaller-treatment.json")
nonparametric <- fromJSON(file="../data/Liver-Day12.5-H3K27ac.raw.json")$Parameters

x <- nonparametric$X

pi <- mixture$Parameters
d1 <- parse.distribution(mixture$Distributions[[1]])
d2 <- parse.distribution(mixture$Distributions[[2]])
d3 <- parse.distribution(mixture$Distributions[[3]])
d4 <- parse.distribution(mixture$Distributions[[4]])
bg <- function(x) pi[1]*d1(x) + pi[2]*d2(x) + pi[3]*d3(x)
fg <- function(x) pi[4]*d4(x)

col=c("darkolivegreen3", "darkolivegreen4", "darkolivegreen4", "firebrick4")

## -----------------------------------------------------------------------------

plot(Y ~ X, nonparametric, type="l", ylab="log density")
points(x, log(pi[1]*d1(x)), col=col[1], lwd=2)
lines (x, log(pi[2]*d2(x)), col=col[2], lwd=2)
lines (x, log(pi[3]*d3(x)), col=col[3], lwd=2)
lines (x, log(pi[4]*d4(x)), col=col[4], lwd=2)
legend("topright", legend=c("Background 1", "Background 2", "Background 3", "Foreground"), col=col, lwd=2, bty="n", pch=c(1,NaN,NaN,NaN), lty=c(NaN,1,1,1,1))

## -----------------------------------------------------------------------------

plot(Y ~ X, nonparametric, type="l", ylab="log density", xlim=c(0,100))
lines(x, log(bg(x)), col=col[2], lwd=2)
lines(x, log(fg(x)), col=col[4], lwd=2)
legend("topright", legend=c("Background", "Foreground"), col=col[3:4], lwd=2, bty="n")

## -----------------------------------------------------------------------------

plot(Y ~ X, nonparametric, type="l", ylab="log density", xlim=c(0,100))
lines(x, log(bg(x)+fg(x)), col=col[4], lwd=2)

## control
## -----------------------------------------------------------------------------
## -----------------------------------------------------------------------------

mixture       <- fromJSON(file="peakCaller-control.json")
nonparametric <- fromJSON(file="../data/Liver-Day12.5-Control.raw.json")$Parameters

x <- nonparametric$X

pi <- mixture$Parameters
d1 <- parse.distribution(mixture$Distributions[[1]])
d2 <- parse.distribution(mixture$Distributions[[2]])
d3 <- parse.distribution(mixture$Distributions[[3]])
d4 <- parse.distribution(mixture$Distributions[[4]])
d5 <- parse.distribution(mixture$Distributions[[5]])
d6 <- parse.distribution(mixture$Distributions[[6]])
d7 <- parse.distribution(mixture$Distributions[[7]])
d8 <- parse.distribution(mixture$Distributions[[8]])
d9 <- parse.distribution(mixture$Distributions[[9]])
bg <- function(x) pi[1]*d1(x) + pi[2]*d2(x) + pi[3]*d3(x) + pi[4]*d4(x) + pi[5]*d5(x) + pi[6]*d6(x) + pi[7]*d7(x) + pi[8]*d8(x)
fg <- function(x) pi[9]*d9(x)

col=c("darkolivegreen3", "darkolivegreen4", "firebrick4")

## -----------------------------------------------------------------------------

plot(Y ~ X, nonparametric, type="l", ylab="log density", xlim=c(0,100))
points(x, log(pi[1]*d1(x)), col=col[1], lwd=2)
points(x, log(pi[2]*d2(x)), col=col[1], lwd=2)
points(x, log(pi[3]*d3(x)), col=col[1], lwd=2)
points(x, log(pi[4]*d4(x)), col=col[1], lwd=2)
points(x, log(pi[5]*d5(x)), col=col[1], lwd=2)
points(x, log(pi[6]*d6(x)), col=col[1], lwd=2)
points(x, log(pi[7]*d7(x)), col=col[1], lwd=2)
lines (x, log(pi[8]*d8(x)), col=col[2], lwd=2)
lines (x, log(pi[9]*d9(x)), col=col[3], lwd=2)
legend("topright", legend=c("Background 1", "Background 2", "Foreground"), col=col, lwd=2, bty="n", pch=c(1,NaN,NaN), lty=c(NaN,1,1))

## -----------------------------------------------------------------------------

plot(Y ~ X, nonparametric, type="l", ylab="log density", xlim=c(0,100))
lines(x, log(bg(x)), col=col[2], lwd=2)
lines(x, log(fg(x)), col=col[3], lwd=2)
legend("topright", legend=c("Background", "Foreground"), col=col[2:3], lwd=2, bty="n")

## -----------------------------------------------------------------------------

plot(Y ~ X, nonparametric, type="l", ylab="log density", xlim=c(0,100))
lines(x, log(bg(x)+fg(x)), col=col[2], lwd=2)
