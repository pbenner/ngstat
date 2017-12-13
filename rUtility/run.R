
library(gridExtra)

source("~/Source/pshape/Rutility/hmm.R")
source("~/Source/pshape/Rutility/hmm.mixture.R")
source("~/Source/pshape/Rutility/hmm.plot.R")
source("~/Source/pshape/Rutility/hmm.stationary.R")

## -----------------------------------------------------------------------------

hmm <- import.hmm("run.json")

## -----------------------------------------------------------------------------

labels <- c("ATAC","H3K9AC","H3K9ME3","H3K27AC","H3K27ME3","H3K4ME1","H3K4ME2","H3K4ME3")

pdf(file="run.pdf")

grid.newpage()
p <- plot.hmm.summary(hmm, labels=labels, main="HMM Differential 11.5->13.5")
grid.draw(p)

dev.off()

## -----------------------------------------------------------------------------

# grep 'log Pdf' run.log | awk '{print $4}' > run.logPdf.txt

t <- read.table("run.logPdf.txt")$V1
plot(t[20:length(t)], type="l")

## set P0 to stationary distribution
### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------

json <- hmm.set.stationary("run.json")
write(toJSON(json), "run.stationary.json")
