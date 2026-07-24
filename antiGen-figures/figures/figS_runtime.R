## ---------------------------------------------------------------------------
## Supplementary figure: wall-clock runtime of antiGen's data pipeline against
## UShER, over the spike region.
##
## Output: figs/runtime.png, figs_vector/runtime.pdf
## ---------------------------------------------------------------------------

source("R/common.R")

suppressPackageStartupMessages(library(dplyr))

TOOL_LABELS <- c(
  "antiGen-data"  = "antiGen pipeline",
  "usher-sampled" = "UShER (sampled)",
  "usher-classic" = "UShER (classic)"
)

tool_colors <- c(
  "antiGen pipeline" = ANTIGEN_BLUE,
  "UShER (sampled)"  = "#c6bbbb",
  "UShER (classic)"  = "#555560"
)

## usher-classic was not run at N = 1e6, so its line simply stops short.
## (The is.na guard is defensive; every spike row is timed.)
runtime <- read.csv("runtime/results_combined.csv") %>%
  filter(region == "spike", !is.na(wall_s)) %>%
  mutate(tool = factor(TOOL_LABELS[tool], levels = TOOL_LABELS))

p <- ggplot(runtime, aes(x = N, y = wall_s, color = tool, group = tool)) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 2) +
  scale_x_log10() +
  scale_y_log10() +
  scale_color_manual(values = tool_colors) +
  labs(x = "number of sequences (N)", y = "wall-clock time (s)",
       color = NULL, title = "Runtime: antiGen's data pipeline vs. UShER") +
  theme_minimal() +
  theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))

save_figure(p, "runtime", width = 6, height = 4.2)
