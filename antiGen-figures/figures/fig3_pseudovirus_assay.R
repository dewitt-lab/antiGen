## ---------------------------------------------------------------------------
## Figure 3: pseudovirus infectivity assay for antiGen's top predicted XFG
## RBD mutations.
##
## Panel A and B are externally drawn cartoons; C and D are the measured
## particle-normalised relative infectivities.
##
## Output: figs/experimental_results.png, figs_vector/experimental_results.pdf
##         extra_figs/{all,de_novo}_soyoon.pdf  (standalone panels for slides)
## ---------------------------------------------------------------------------

source("R/common.R")

suppressPackageStartupMessages(library(dplyr))

## The replicate points are drawn with geom_jitter, so their horizontal
## offsets come from the RNG. Seed it, or the panels move every run.
set.seed(1)

## Wide, one measurement row. Each column is one well, named
## "<construct>_<mutation>"; the three replicates of a mutation share a name,
## so read.csv appends ".1"/".2". The first 30 columns are the 10 de novo
## mutations, the next 30 the 10 mutations already seen in circulation.
DE_NOVO_ROWS <- 1:30
ALL_ROWS     <- 31:60

raw <- read.csv("experiments/Average Relative RLU_particle.csv")

df <- data.frame(Mutation = colnames(raw), RLU = as.numeric(raw[1, ]))
df$Mutation <- gsub(".*_", "", df$Mutation)              # "X86_K403R" -> "K403R"
df$Mutation <- gsub("..insert..", "ins", df$Mutation)    # "483..insert..C" -> "483insC"
df$Mutation <- gsub("\\..*", "", df$Mutation)            # drop the ".1"/".2" replicate tag
df$pos      <- as.numeric(gsub("\\D", "", df$Mutation))

plot_df <- rbind(
  transform(df[ALL_ROWS, ],     set = "All mutations"),
  transform(df[DE_NOVO_ROWS, ], set = "De novo mutations")
)

## Shared y-limit across both panels, covering points as well as mean + SD.
panel_top <- plot_df %>%
  group_by(set, Mutation) %>%
  summarise(top = mean(RLU) + sd(RLU), .groups = "drop")
ylim_range <- c(0, max(c(plot_df$RLU, panel_top$top), na.rm = TRUE) * 1.05)

make_panel <- function(d, s, fill, ylim_range, y_title = TRUE) {
  sub  <- d[d$set == s, ]
  summ <- sub %>%
    group_by(Mutation, pos) %>%
    summarise(mean_RLU = mean(RLU), sd_RLU = sd(RLU), .groups = "drop")

  ## Order mutations within this panel by residue position.
  lev <- unique(summ$Mutation[order(summ$pos)])
  summ$Mutation <- factor(summ$Mutation, levels = lev)
  sub$Mutation  <- factor(sub$Mutation,  levels = lev)

  p <- ggplot() +
    geom_col(data = summ, aes(x = Mutation, y = mean_RLU),
             fill = fill, width = 0.7, alpha = 0.85) +
    geom_errorbar(data = summ,
                  aes(x = Mutation, ymin = mean_RLU - sd_RLU, ymax = mean_RLU + sd_RLU),
                  width = 0.3, color = "grey30") +
    geom_jitter(data = sub, aes(x = Mutation, y = RLU),
                width = 0.12, height = 0, size = 3,
                shape = 21, fill = "white", color = "grey20", stroke = 0.75) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey60") +
    coord_cartesian(ylim = ylim_range) +
    labs(x = NULL, y = "Relative infectivity\n(RLU/particle rel. XFG)") +
    ggtitle(s) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  if (!y_title) p <- p + theme(axis.title.y = element_blank())
  p
}

p_all <- make_panel(plot_df, "All mutations", "#38761d", ylim_range, y_title = TRUE)
p_dn  <- make_panel(plot_df, "De novo mutations", "#6aa84f", ylim_range, y_title = FALSE) +
  theme(plot.title = element_text(hjust = 0.07))

figure <- plot_grid(
  plot_cartoon("cartoons/assay_schematic.png",
               "Overview of pseudovirus infectivity assay"),
  plot_grid(
    plot_cartoon("cartoons/mutation_map.png", "Locations of mutations") +
      theme(plot.title = element_text(hjust = 0.2)),
    p_all,
    p_dn,
    labels = LETTERS[2:4],
    nrow = 1,
    rel_widths = c(3, 2, 2)
  ),
  ncol = 1,
  labels = c("A", ""),
  rel_heights = c(4.8, 2)
)

save_figure(figure, "experimental_results", width = 9, height = 7)

## Standalone versions of the two data panels, for slides.
ggsave("extra_figs/all_soyoon.pdf",     p_all, width = 4, height = 2)
ggsave("extra_figs/de_novo_soyoon.pdf", p_dn,  width = 4, height = 2)
