
library(gridExtra)
library(reshape2)
library(ggplot2)

## -----------------------------------------------------------------------------

plot.hmm.matrix <- function(m, ...) {
    p <- ggplot(data = melt(m), aes(Var2, Var1, fill = value))+
        geom_tile(color = "white")+
        scale_fill_gradientn(colors = c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"),
                             limit = c(0,1), guide=FALSE) +
        coord_fixed()
    p <- p + 
        theme_minimal() +
        theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.border     = element_blank(),
            panel.background = element_blank(),
            axis.ticks = element_blank())
    p
}

plot.hmm.summary <- function(hmm, labels=NULL, main="", states.nrow=NULL, states.ncol=NULL) {
    if (class(hmm$EDist) == "list") {
        m        <- hmm$Tr
        labels.x <- hmm$StateMap
        labels.y <- hmm$StateMap
        p1 <- plot.hmm.matrix(m) +
            scale_x_continuous(breaks = 1:ncol(m), labels=labels.x, position = "top") +
            scale_y_reverse   (breaks = 1:nrow(m), labels=labels.y) +
            ggtitle(main)
        p2 <- list()
        for (m in hmm$EDist) {
            p2[[length(p2)+1]] <- plot.hmm.matrix(t(m)) +
                                 scale_y_reverse(breaks = 1:ncol(m), labels=labels) +
                                 ggtitle(sprintf("State %d", hmm$StateMap[length(p2)+1]))
        }
        p2 <- arrangeGrob(grobs = p2, nrow=states.nrow, ncol=states.ncol)
        arrangeGrob(p1, p2, ncol=2)
    } else {
        if (is.null(hmm$EDist)) {
            m <- hmm$Tr
            labels.x <- hmm$StateMap
            labels.y <- hmm$StateMap
        } else {
            edist <- normalize.matrix(as.matrix(hmm$EDist[hmm$StateMap+1,]), bycol=TRUE)
            ## construct matrix
            na    <- matrix(NA, ncol(edist), ncol(edist))
            tr    <- hmm$Tr
            m1    <- cbind(na, NA, t(edist))
            m2    <- cbind(edist, NA, tr)
            m     <- rbind(m1, NA, m2)
            ## define labels
            if (is.null(labels)) {
                labels <- rep("", ncol(edist))
            }
            labels.x <- c(rep("", length(labels)+1), hmm$StateMap)
            labels.y <- c(labels, "", hmm$StateMap)
        }
        p <- plot.hmm.matrix(m) +
            scale_x_continuous(breaks = 1:ncol(m), labels=labels.x, position = "top") +
            scale_y_reverse   (breaks = 1:nrow(m), labels=labels.y) +
            ggtitle(main)
        arrangeGrob(p)
    }
}
