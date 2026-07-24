## ---------------------------------------------------------------------------
## Supplementary figure: next-mutation recall AUC on the extended ("more
## data") SARS-CoV-2 evaluation, the only one that includes CoVFit.
##
## Output: figs/supp_covfit.png, figs_vector/supp_covfit.pdf
## ---------------------------------------------------------------------------

source("R/common.R")

EVAL_DIR <- "data/more_data/simple"

## After the MSA Transformer rows are dropped, each evaluation file holds
## these eight models in this order. The DMS rows are relabelled for display;
## the rest keep the names the evaluation wrote.
MODEL_LABELS <- c(
  "antiGen",                 # written as "trained"
  "PSSM",                    # written as "pssm_root"
  "CoVFit",
  "EVEscape",
  "BA.1 DMS (bind)",         # written as "Omicron_BA1_bind", etc.
  "BA.1 DMS (expression)",
  "BA.2 DMS (bind)",
  "BA.2 DMS (expression)"
)
MODEL_TYPES <- c(
  "antiGen",
  "Pandemic Data in Training Window",   # PSSM
  "Pandemic Data in Training Window",   # CoVFit
  "Pre-Pandemic Data Only",             # EVEscape
  rep("Deep Mutational Scans", 4)
)
stopifnot(all(MODEL_TYPES %in% MODEL_TYPE_LEVELS))

RAW_MODELS <- c("trained", "pssm_root", "CoVFit", "EVEscape",
                "Omicron_BA1_bind", "Omicron_BA1_expr",
                "Omicron_BA2_bind", "Omicron_BA2_expr")

#' Read one quantile-hit-rate file and reduce it to one recall AUC per model.
read_recall_auc <- function(file) {
  df <- read.csv(file)
  df <- df[!grepl("MSA", df$model), ]
  stopifnot(identical(df$model, RAW_MODELS))

  ## Trapezoidal AUC over q = 0, 0.1, ..., 1. The file holds the nine
  ## interior quantiles; the curve is anchored at (0, 0) and (1, 1).
  quantiles <- as.matrix(df[, -1])
  auc <- apply(quantiles, 1, function(vals) {
    sum((c(vals, 1) + c(0, vals)) / 2 * 0.1)
  })

  data.frame(model = MODEL_LABELS, auc = as.numeric(auc), type = MODEL_TYPES)
}

make_auc_plot <- function(df, title, model_levels, xmin = 0) {
  df$model <- factor(df$model, levels = model_levels)

  ## Medal text for the top three models in this panel.
  df$medal <- c("1st", "2nd", "3rd")[rank(-df$auc, ties.method = "first")]
  top3 <- df[!is.na(df$medal), ]

  ggplot(df, aes(y = model, x = auc, fill = type)) +
    geom_col() +
    geom_text(
      data = top3,
      aes(x = auc, y = model, label = medal),
      inherit.aes = FALSE,
      hjust = 1.1, size = 2.5, fontface = "bold", color = "white"
    ) +
    scale_fill_manual(values = MODEL_TYPE_COLORS) +
    coord_cartesian(xlim = c(xmin, max(df$auc))) +
    theme_minimal() +
    theme(legend.position = "top", axis.title.y = element_blank()) +
    labs(fill = "Model type", x = "AUC") +
    ggtitle(title)
}

auc_all    <- read_recall_auc(find_metric_file(EVAL_DIR, "quantile_hit_rates_all"))
auc_unseen <- read_recall_auc(find_metric_file(EVAL_DIR, "quantile_hit_rates_unseen"))

## Shared bar order, grouped by model type and identical in both panels.
## Levels run bottom-to-top on the y-axis, so antiGen ends up on top. The
## sort is stable, so file order is preserved within each type.
type_order <- c("Deep Mutational Scans",
                "Pandemic Data in Training Window",
                "Pre-Pandemic Data Only",
                "antiGen")
model_levels <- auc_all$model[order(factor(auc_all$type, levels = type_order))]

p_all    <- make_auc_plot(auc_all,    "Next-mutation recall, all mutations", model_levels)
p_unseen <- make_auc_plot(auc_unseen, "Next-mutation recall, de novo",       model_levels)

figure <- plot_grid(
  extract_legend(p_all),
  plot_grid(
    p_all    + theme(legend.position = "none"),
    p_unseen + theme(legend.position = "none"),
    labels = c("A", "B"),
    nrow   = 1
  ),
  ncol = 1,
  rel_heights = c(0.12, 1)
)

save_figure(figure, "supp_covfit", width = 9, height = 2.2)
