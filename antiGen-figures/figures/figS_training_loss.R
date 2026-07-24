## ---------------------------------------------------------------------------
## Supplementary figure: training loss curves for antiGen and the fine-tuned
## protein language models on SARS-CoV-2.
##
## Output: figs/loss.png, figs_vector/loss.pdf
## ---------------------------------------------------------------------------

source("R/common.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

raw <- read.csv("loss/training_loss_curves_sc2.csv", check.names = FALSE)

## Map each run's column name onto its base model and fine-tuning mode.
runs <- tibble::tribble(
  ~col,                                            ~model,         ~mode,
  "simple_d128_n2_lr0.0001",                       "antiGen",      "Single",
  "esm2_esm2_t12_35M_UR50D_lr0.0001_partial4",     "ESM-2 (35M)",  "Partial",
  "esm2_esm2_t12_35M_UR50D_lr0.001_frozen",        "ESM-2 (35M)",  "Frozen",
  "esm2_esm2_t12_35M_UR50D_lr1e-05_full",          "ESM-2 (35M)",  "Full",
  "esm2_esm2_t33_650M_UR50D_lr0.0001_partial4",    "ESM-2 (650M)", "Partial",
  "esm2_esm2_t33_650M_UR50D_lr0.001_frozen",       "ESM-2 (650M)", "Frozen",
  "esm2_esm2_t33_650M_UR50D_lr1e-05_full",         "ESM-2 (650M)", "Full",
  "eve_ft_lr0.0001",                               "EVE",          "Single",
  "tranception_ft_both_lr0.0001",                  "Tranception",  "Single"
)

loss <- raw %>%
  pivot_longer(-epoch, names_to = "col", values_to = "loss") %>%
  left_join(runs, by = "col") %>%
  mutate(
    model = factor(model, levels = c("antiGen", "ESM-2 (35M)", "ESM-2 (650M)",
                                     "EVE", "Tranception")),
    mode  = factor(mode,  levels = c("Single", "Full", "Partial", "Frozen"))
  )

## NOTE: the two ESM-2 keys below are spelled "ESM 2" while the factor levels
## above are spelled "ESM-2", so those two levels fall through to the
## unmatched-value default: their curves are drawn grey and they are dropped
## from the colour legend. That is how the published figure looks, so the
## mismatch is kept deliberately -- aligning the names would change it.
model_colors <- c(
  "antiGen"      = ANTIGEN_BLUE,
  "ESM 2 (35M)"  = "#ccccbb",
  "ESM 2 (650M)" = "#bba3bb",
  "EVE"          = "#99aaaa",
  "Tranception"  = "#999977"
)

## Single-mode models render solid (same as Full), but only the three real
## fine-tuning modes appear in the legend.
mode_lty <- c("Single" = "solid", "Full" = "solid",
              "Partial" = "dashed", "Frozen" = "dotted")

p <- ggplot(loss, aes(epoch, loss, color = model, linetype = mode, group = col)) +
  geom_line(linewidth = 0.7) +
  # Redraw antiGen thicker so it stays legible against the fine-tune bundle.
  geom_line(data = filter(loss, model == "antiGen"),
            linewidth = 1.3, show.legend = FALSE) +
  scale_color_manual(values = model_colors) +
  scale_linetype_manual(values = mode_lty,
                        breaks = c("Full", "Partial", "Frozen")) +
  labs(x = "Epoch", y = "Training loss (NLL)",
       color = "Base model", linetype = "Fine-tuning mode",
       title = "Training loss curves (SARS-CoV-2)") +
  theme_minimal() +
  theme(legend.position = "right",
        legend.key.width = unit(1.5, "lines"))

save_figure(p, "loss", width = 6, height = 4.2)
