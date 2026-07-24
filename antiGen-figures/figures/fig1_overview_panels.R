## ---------------------------------------------------------------------------
## Figure 1 component panels. Figure 1 itself is assembled outside R (the
## final artwork lives in figs_vector/overview.svg), so this script only emits
## the individual panels into fig1/:
##
##   pred.{png,pdf}        antiGen's predicted mutation matrix over the RBD
##   obs.{png,pdf}         the same region of the observed PSSM
##   <virus>.json_tree.png a time-scaled phylogeny per virus in trees/
##
## The recall inset (fig1/auc.png) comes from fig2_sc2_benchmarks.R, which
## already has the evaluation curves loaded.
## ---------------------------------------------------------------------------

source("R/common.R")

library(reshape2)
library(jsonlite)

PANEL_DIR <- "fig1"

## Only the first stretch of the RBD is shown; the panel is a narrow strip in
## the assembled figure, not the full domain.
HEATMAP_LAST_RESIDUE <- 380

## Heatmap cells are drawn on a continuous fill scale, so exact ties would be
## ordered arbitrarily; a fixed seed keeps the tie-breaking jitter (below)
## reproducible.
set.seed(1)

ref <- strsplit(readLines("ref/BA.2.txt", warn = FALSE), "")[[1]]


## --- Mutation-matrix heatmaps ----------------------------------------------

#' Heatmap of one residue x amino-acid matrix over the RBD.
#'
#' Wild-type cells are blanked, so the panel shows mutations only. Axis text
#' and ticks are suppressed because the panel gets its labels during figure
#' assembly.
mtx_to_heatmap <- function(mat) {
  mat <- mask_wildtype(mat, ref)
  mat <- mat[RBD_RANGE, 1:20]

  df <- melt(as.matrix(mat), varnames = c("row", "col"))
  df$row <- as.integer(df$row)          # residue number, from the row names
  df$col <- factor(df$col, levels = colnames(mat))

  ## Break exact ties (e.g. structural zeros) so neighbouring cells do not
  ## render as one flat block.
  df$value <- df$value + rnorm(length(df$value), 0, 0.0000001)

  df <- df[df$row <= HEATMAP_LAST_RESIDUE, ]

  ggplot(df, aes(x = row, y = col, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "red", na.value = "white") +
    scale_x_continuous(breaks = seq(340, 430, by = 20), expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    labs(x = "Residue", y = "Amino acid") +
    theme_minimal() +
    coord_fixed() +
    theme(panel.grid  = element_blank(),
          legend.position = "none",
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
}

predicted <- read.csv(file.path(
  "data/sc2/simple",
  "variant_root_predictions_simple_d128_n2_lr0.0001_20260426_183918.csv"
))
## Observed counts are heavily skewed; log1p makes the low-count cells visible.
observed <- log(1 + read.csv("pssms/time_5.9.csv")[, -1])

for (panel in list(list(name = "pred", mat = predicted),
                   list(name = "obs",  mat = observed))) {
  p <- mtx_to_heatmap(panel$mat)
  ggsave(file.path(PANEL_DIR, paste0(panel$name, ".png")), p, width = 3, height = 1.5)
  ggsave(file.path(PANEL_DIR, paste0(panel$name, ".pdf")), p, width = 6, height = 3)
}


## --- Time-scaled phylogenies ----------------------------------------------

## Plain black rectangular phylogeny from a Nextstrain v2 JSON. x is the
## decimal date from node_attrs$num_date$value; branch thickness scales with
## the number of descendant tips, Nextstrain style, on both the horizontal
## branches and the vertical connectors.

#' Rectangular coordinates for every branch of a Nextstrain tree.
#'
#' Tips take successive integer y positions in traversal order, internal nodes
#' sit at the midpoint of their descendants. Returns the horizontal branches
#' and the vertical connectors, each carrying the tip count `n` of its clade.
tree_segments <- function(root) {
  ytip <- 0L
  h_x <- h_xend <- h_y <- h_n <- numeric(0)   # horizontal branches
  v_x <- v_y <- v_yend <- v_n <- numeric(0)   # vertical connectors

  walk <- function(node, parent_x) {
    date <- node$node_attrs$num_date$value
    if (is.null(date)) date <- parent_x       # fallback, shouldn't trigger
    children <- node$children

    if (is.null(children)) {                  # tip
      ytip <<- ytip + 1L
      y <- ytip
      n <- 1
    } else {                                  # internal node
      res <- vapply(children, function(ch) walk(ch, date), numeric(2))
      cy <- res[1, ]
      y  <- mean(range(cy))
      n  <- sum(res[2, ])                     # tips in this clade
      v_x    <<- c(v_x, date)
      v_y    <<- c(v_y, min(cy))
      v_yend <<- c(v_yend, max(cy))
      v_n    <<- c(v_n, n)
    }

    if (!is.null(parent_x)) {                 # branch down from the parent
      h_x    <<- c(h_x, parent_x)
      h_xend <<- c(h_xend, date)
      h_y    <<- c(h_y, y)
      h_n    <<- c(h_n, n)
    }
    c(y, n)
  }

  invisible(walk(root, NULL))

  list(
    horizontal = data.frame(x = h_x, xend = h_xend, y = h_y, n = h_n),
    vertical   = data.frame(x = v_x, y = v_y, yend = v_yend, n = v_n),
    n_tips     = ytip
  )
}

for (file in list.files("trees")) {
  seg <- tree_segments(fromJSON(file.path("trees", file), simplifyVector = FALSE)$tree)
  h <- seg$horizontal
  v <- seg$vertical

  p <- ggplot() +
    geom_segment(data = v, aes(x = x, xend = x, y = y, yend = yend, linewidth = n),
                 lineend = "round") +
    geom_segment(data = h, aes(x = x, xend = xend, y = y, yend = y, linewidth = n),
                 lineend = "round") +
    scale_linewidth_continuous(range = c(0.1, 1.6), trans = "sqrt", guide = "none") +
    scale_x_continuous(name = "Date", breaks = scales::pretty_breaks(8)) +
    xlim(c(NA, max(v$x))) +
    xlab("Date") +
    theme_minimal(base_size = 12) +
    theme(
      axis.title.y       = element_blank(),
      axis.text.y        = element_blank(),
      axis.ticks.y       = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank(),
      panel.grid.minor.x = element_blank()
    )

  ggsave(file.path(PANEL_DIR, paste0(file, "_tree.png")), p, width = 5, height = 3)
  cat(file, "- tips:", seg$n_tips, " branches:", nrow(h), "\n")
}
