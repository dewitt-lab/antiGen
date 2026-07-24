## ---------------------------------------------------------------------------
## Figure 4 and its supplement: antiGen generalises to three endemic viruses.
##
## Main text  (endemic_results):      recall AUC bars + PSSM precision curves,
##                                    two metrics each.
## Supplement (supp_endemic_results): all four metrics of both families, as
##                                    curves.
##
## Output: figs/endemic_results.png,      figs_vector/endemic_results.pdf
##         figs/supp_endemic_results.png, figs_vector/supp_endemic_results.pdf
## ---------------------------------------------------------------------------

source("R/common.R")

library(reshape2)

apply_paper_defaults()

## antiGen, DMS, EVE, EVmutation-style PSSM baselines. The palette is indexed
## positionally below, so keep the order.
my_colors <- c("#555560", "#999977", ANTIGEN_BLUE, "#888f88", "#ccccaa", "#c6bbbb")

LONGFORM_HIT_RATE_METRICS <- c(
  "All mutations",
  "De novo",
  "All mutations + gap/stop mask",
  "De novo + gap/stop mask"
)

LONGFORM_VARIANT_EMERGENCE_METRICS <- c(
  "All mutations, 1+ cases",
  "De novo, 1+ cases",
  "All mutations, 10+ cases",
  "De novo, 10+ cases"
)

viruses        <- c("flu", "rsv-a", "denv2")
viruses_formal <- c("H3N2 influenza, HA protein", "RSV A, F protein",
                    "DENV2, E protein")

## Baselines shared by all three viruses, plus the per-virus DMS assay where
## one exists.
SHARED_MODELS <- c("simple_d128_n2_lr0.0001", "pssm_root", "pssm_root_position",
                   "eve_frozen", "blosum62")
DMS_MODEL <- c("rsv-a" = "cell.entry", "flu" = "HK19_DMS")

## Display names; "trained" is renamed to "antiGen" later, once the palette
## has been keyed off it.
MODEL_LABELS <- c(
  "simple_d128_n2_lr0.0001" = "trained",
  "pssm_root"               = "PSSM",
  "pssm_root_position"      = "PSSM Conservation",
  "eve_frozen"              = "EVE",
  "blosum62"                = "BLOSUM62",
  "cell.entry"              = "DMS",
  "HK19_DMS"                = "DMS"
)


## --- Load evaluation curves ------------------------------------------------

read_curves <- function(virus) {
  eval_dir <- file.path("data", virus, "evals")
  # DENV2 has no DMS assay, so nothing is appended for it.
  valid_models <- c(SHARED_MODELS,
                    if (virus %in% names(DMS_MODEL)) DMS_MODEL[[virus]])

  recall <- lapply(HIT_RATE_METRICS, function(metric) {
    df <- read.csv(find_metric_file(eval_dir, metric))
    ## Recall is 0 at q = 0 and 1 at q = 1 by construction; the evaluation
    ## only writes the nine interior quantiles.
    df$top_0.0.   <- 0
    df$top_100.0. <- 1
    df <- melt(df[df$model %in% valid_models, ], id.vars = "model")
    df$variable <- as.numeric(gsub("\\..*", "", gsub("top_", "", df$variable))) / 100
    colnames(df)[2] <- "x"
    df$metric <- metric
    df$virus  <- virus
    df
  })

  precision <- lapply(VARIANT_EMERGENCE_METRICS, function(metric) {
    df <- read.csv(find_metric_file(eval_dir, metric))
    df <- df[df$model %in% valid_models & df$top_quantile == 0.1, -2]
    df <- melt(df, id.vars = "model")
    colnames(df)[2] <- "x"
    df$x      <- as.numeric(gsub("t_", "", df$x))
    df$metric <- metric
    df$virus  <- virus
    df
  })

  do.call(rbind, c(recall, precision))
}

performance_by_x <- do.call(rbind, lapply(viruses, read_curves))
performance_by_x$model <- unname(MODEL_LABELS[performance_by_x$model])


## --- Figure assembly -------------------------------------------------------

## One shared palette across all viruses. DENV2 has no DMS assay, so when the
## DMS series is absent the fourth colour is dropped to keep every other
## model on the colour it has in the other panels.
palette_for <- function(df) {
  models <- unique(ifelse(df$model == "trained", "antiGen", df$model))
  if ("DMS" %in% models) {
    setNames(my_colors, models)
  } else {
    setNames(my_colors[-4], models)
  }
}

#' One line panel: every model's curve for one metric and one virus.
make_curve_plot <- function(df, palette, metric, virus, xlab, ylab, title) {
  sub <- df[df$virus == virus & df$metric == metric, ]
  sub$model <- ifelse(sub$model == "trained", "antiGen", sub$model)

  ## Emergence curves start flat at zero until the first variant appears;
  ## drop that dead lead-in so the interesting range fills the panel.
  all_zero_per_x <- tapply(sub$value, sub$x, function(v) all(v == 0))
  zero_xs <- as.numeric(names(all_zero_per_x))[all_zero_per_x]
  if (length(zero_xs) > 0) sub <- sub[sub$x >= max(zero_xs), ]

  ggplot(sub, aes(x = x, y = value, color = model, group = model,
                  linetype = (model != "antiGen"))) +
    geom_line(linewidth = 0.75) +
    scale_color_manual(values = palette) +
    scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "21"),
                          guide = "none") +
    xlab(xlab) + ylab(ylab) +
    ggtitle(title) +
    theme_paper() + labs(color = NULL) +
    theme(legend.position = "none")
}

#' Same data as a bar chart of each curve's area, with the winner medalled.
make_auc_plot <- function(df, palette, metric, virus, xlab, title, xmin = 0) {
  sub <- df[df$virus == virus & df$metric == metric, ]
  sub$model <- ifelse(sub$model == "trained", "antiGen", sub$model)

  models_here <- unique(sub$model)
  auc_vals <- sapply(models_here, function(m) {
    d  <- sub[sub$model == m, ]
    o  <- order(d$x)
    xx <- d$x[o]; yy <- d$value[o]
    sum(diff(xx) * (head(yy, -1) + tail(yy, -1)) / 2)
  })
  auc <- data.frame(model = models_here, value = as.numeric(auc_vals),
                    stringsAsFactors = FALSE)

  ## antiGen on top of the y-axis (last factor level); others below in
  ## whatever order they appear.
  auc$model <- factor(auc$model,
                      levels = c(setdiff(auc$model, "antiGen"), "antiGen"))

  winner <- auc[rank(-auc$value, ties.method = "first") == 1, ]

  ggplot(auc, aes(y = model, x = value, fill = model)) +
    geom_col() +
    geom_text(
      data = winner,
      aes(x = value, y = model, label = "1st"),
      inherit.aes = FALSE,
      hjust = 1.1, size = 2.5, fontface = "bold", color = "white"
    ) +
    scale_fill_manual(values = palette) +
    coord_cartesian(xlim = c(xmin, max(auc$value))) +
    xlab(xlab) +
    ggtitle(title) +
    theme_paper() +
    theme(
      legend.position = "none",
      axis.title.y    = element_blank(),
      axis.text.y     = element_blank(),
      axis.ticks.y    = element_blank(),
      plot.title      = element_text(hjust = 0.5)
    )
}

#' Stack one block of rows per `section`, with viruses across the columns.
#'
#' Each section is a list of: metrics, longform_metrics, panel_title, xlab,
#' ylab, and optionally as_auc / auc_xlab to draw AUC bars instead of curves.
plot_virus_grid_stacked <- function(df, sections) {
  n_cols <- length(viruses)
  rows_per_section <- sapply(sections, function(s) length(s$metrics))
  letter_offsets <- c(0, cumsum(rows_per_section * n_cols))
  palette <- palette_for(df)

  ## Shared horizontal legend, built from data pooled across all viruses so
  ## that every model appears in it.
  legend_sub <- df[df$metric == sections[[1]]$metrics[1], ]
  legend_sub$model <- ifelse(legend_sub$model == "trained", "antiGen", legend_sub$model)
  shared_legend <- extract_legend(
    ggplot(legend_sub, aes(x = x, y = value, color = model,
                           group = interaction(model, virus))) +
      geom_line(linewidth = 0.75) +
      scale_color_manual(values = palette) +
      theme_paper() + labs(color = NULL) +
      guides(color = guide_legend(nrow = 1)) +
      theme(
        legend.position      = "top",
        legend.direction     = "horizontal",
        legend.justification = "center",
        legend.key.width     = unit(1, "lines"),
        legend.key.height    = unit(0.6, "lines"),
        legend.margin        = margin(0, 0, 0, 0)
      )
  )

  ## Virus column headers: italic name over a rule.
  header_row <- plot_grid(
    NULL,
    plot_grid(plotlist = lapply(seq_along(viruses), function(i) {
      ggdraw() +
        draw_label(viruses_formal[i], fontface = "italic", size = 13) +
        draw_line(x = c(0.05, 0.95), y = c(0.02, 0.02),
                  color = "black", linewidth = 0.5)
    }), nrow = 1),
    nrow = 1,
    rel_widths = c(0.04, 1)
  )

  section_blocks <- lapply(seq_along(sections), function(s) {
    sec <- sections[[s]]
    base_letter <- letter_offsets[s]

    row_labels <- lapply(sec$longform_metrics, function(m) {
      m <- gsub(" \\+ ", "\n", m)
      m <- gsub(", ", "\n", m)
      ggdraw() + draw_label(m, fontface = "italic", size = 13, angle = 90)
    })

    section_rows <- lapply(seq_along(sec$metrics), function(i) {
      row_plots <- lapply(viruses, function(v) {
        if (isTRUE(sec$as_auc)) {
          make_auc_plot(df, palette, sec$metrics[i], v,
                        xlab = sec$auc_xlab, title = sec$panel_title)
        } else {
          make_curve_plot(df, palette, sec$metrics[i], v,
                          sec$xlab, sec$ylab, sec$panel_title)
        }
      })

      plot_grid(
        row_labels[[i]],
        plot_grid(plotlist = row_plots, nrow = 1,
                  labels = LETTERS[base_letter + (i - 1) * n_cols + seq_len(n_cols)]),
        nrow = 1,
        rel_widths = c(0.04, 1)
      )
    })

    plot_grid(plotlist = section_rows, ncol = 1)
  })

  ## Bar sections are half the height of curve sections. The legend is
  ## repeated directly under the first section when that section is bars,
  ## because the bars carry no axis labels of their own.
  is_auc <- sapply(sections, function(s) isTRUE(s$as_auc))
  section_heights <- ifelse(is_auc, rows_per_section * 0.5, rows_per_section)

  comps   <- list(header_row)
  heights <- 0.15
  for (s in seq_along(section_blocks)) {
    comps   <- c(comps, list(section_blocks[[s]]))
    heights <- c(heights, section_heights[s])
    if (s == 1 && isTRUE(is_auc[1])) {
      comps   <- c(comps, list(shared_legend))
      heights <- c(heights, 0.15)
    }
  }
  comps   <- c(comps, list(shared_legend))
  heights <- c(heights, 0.15)

  plot_grid(plotlist = comps, ncol = 1, rel_heights = heights)
}


## --- Emit both figures -----------------------------------------------------

build_figure <- function(supp) {
  ## Supplement: every panel is a curve, all four metrics per family.
  ## Main text: recall as AUC bars, precision as curves, two metrics each.
  idx <- if (supp) 1:4 else 1:2

  plot_virus_grid_stacked(performance_by_x, list(
    list(
      metrics          = HIT_RATE_METRICS[idx],
      longform_metrics = LONGFORM_HIT_RATE_METRICS[idx],
      xlab             = "quantile (q)",
      ylab             = if (supp) "Recall" else "P(obs. mut. in top q-quantile)",
      panel_title      = "Next-mutation recall",
      as_auc           = !supp,
      auc_xlab         = "Recall AUC"
    ),
    list(
      metrics          = VARIANT_EMERGENCE_METRICS[idx],
      longform_metrics = LONGFORM_VARIANT_EMERGENCE_METRICS[idx],
      xlab             = if (supp) "Date (t, years into testing window)"
                         else      "Date (t, years into pandemic)",
      ylab             = if (supp) "Precision" else "P(mut. in top decile obs. by t)",
      panel_title      = "PSSM precision"
    )
  ))
}

save_figure(build_figure(supp = FALSE), "endemic_results",      width = 9, height = 9)
save_figure(build_figure(supp = TRUE),  "supp_endemic_results", width = 9, height = 13)
