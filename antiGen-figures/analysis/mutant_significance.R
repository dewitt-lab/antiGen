## ---------------------------------------------------------------------------
## Statistics reported in the text, not a figure.
##
## Of antiGen's top 10 predicted RBD mutations, how many improve ACE2 binding
## or spike expression in the KP.3 and LP.8 deep mutational scans -- and how
## surprising is that count, given each scan's overall rate of beneficial
## mutations? One-sided binomial tail probabilities.
##
## Run from the project root:  Rscript analysis/mutant_significance.R
## ---------------------------------------------------------------------------

## Observed hits out of the 10 predictions, for all mutations and for the
## de novo subset.
N_PREDICTIONS <- 10
HITS_ALL      <- 6
HITS_DE_NOVO  <- 4

dms <- read.csv("dms/latest.txt")

## Root PSSM over the training window: a zero count means the mutation had
## never been observed, i.e. it is de novo. `pos` is 0-based; the columns for
## stop and gap arrive from read.csv as "X." and "X..1".
pssm <- read.csv("dms/root.csv")
pssm$pos <- pssm$pos + 1
colnames(pssm)[22:23] <- c("*", "-")

dms$new <- vapply(seq_len(nrow(dms)), function(row) {
  pssm[dms$position[row], match(dms$mutant[row], colnames(pssm))] == 0
}, logical(1))

#' P(>= hits successes in N_PREDICTIONS draws) at the scan's background rate.
binom_tail <- function(effects, hits) {
  1 - pbinom(hits - 1, N_PREDICTIONS, mean(effects > 0, na.rm = TRUE))
}

for (target in c("Omicron_KP3", "Omicron_LP8")) {
  sub <- dms[dms$target == target, ]
  for (subset in c("all mutations", "de novo")) {
    d <- if (subset == "all mutations") sub else sub[sub$new, ]
    hits <- if (subset == "all mutations") HITS_ALL else HITS_DE_NOVO
    cat(sprintf("%-12s %-14s  bind p = %.4g   expr p = %.4g\n",
                target, subset,
                binom_tail(d$delta_bind, hits),
                binom_tail(d$delta_expr, hits)))
  }
}

## Size of the training set at the 1.338-year train/test cutoff, quoted
## alongside the gating results.
pssm_cutoff <- read.csv("pssm_by_cutoff/root_1.338.csv")[, -1]
cat("\ncutoff 1.338: ", sum(pssm_cutoff > 0) - nrow(pssm_cutoff),
    " distinct mutations, ", rowSums(pssm_cutoff)[1], " sequences\n", sep = "")
