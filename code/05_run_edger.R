#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  if (!requireNamespace("edgeR", quietly = TRUE)) {
    stop("The R package 'edgeR' is required. Install with BiocManager::install('edgeR').")
  }
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 18) {
  stop(
    "Usage: Rscript 05_run_edger.R ",
    "--counts counts.csv --metadata metadata.csv --comparisons grp1,grp2;grp3,grp4 ",
    "--region whole_brain --cell-type CellType --min-cells 30 --covariates none ",
    "--out-dir results/edger --plot-dir plots/edgeR"
  )
}

get_arg <- function(flag) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) stop(paste("Missing argument:", flag))
  args[[idx + 1]]
}

parse_comparisons <- function(text) {
  pairs <- strsplit(text, ";", fixed = TRUE)[[1]]
  out <- lapply(pairs, function(pair) {
    fields <- strsplit(pair, ",", fixed = TRUE)[[1]]
    if (length(fields) != 2) stop(paste("Bad comparison:", pair))
    fields
  })
  out
}

safe_name <- function(x) {
  gsub("[^A-Za-z0-9_.-]+", "_", x)
}

save_static_plot <- function(path_prefix, width, height, draw) {
  png(paste0(path_prefix, ".png"), width = width, height = height, res = 150)
  draw()
  dev.off()
  svg(paste0(path_prefix, ".svg"), width = width / 150, height = height / 150)
  draw()
  dev.off()
}

counts_path <- get_arg("--counts")
metadata_path <- get_arg("--metadata")
comparisons <- parse_comparisons(get_arg("--comparisons"))
region <- get_arg("--region")
cell_type <- get_arg("--cell-type")
min_cells <- as.integer(get_arg("--min-cells"))
covariates_text <- get_arg("--covariates")
out_dir <- get_arg("--out-dir")
plot_dir <- get_arg("--plot-dir")
covariates <- if (covariates_text %in% c("", "none", "None", "NONE")) character(0) else strsplit(covariates_text, ",", fixed = TRUE)[[1]]

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

counts <- read.csv(counts_path, row.names = 1, check.names = FALSE)
metadata <- read.csv(metadata_path, row.names = 1, check.names = FALSE)

metadata_all <- metadata[
  metadata$region == region &
    metadata$cell_type == cell_type &
    metadata$n_cells >= min_cells,
  ,
  drop = FALSE
]

if (nrow(metadata_all) < 4) stop("Need at least four pseudobulk samples after filtering.")

counts <- counts[rownames(metadata_all), , drop = FALSE]
counts_mat <- t(as.matrix(counts))
storage.mode(counts_mat) <- "integer"

group <- factor(metadata_all$group)
group_levels <- levels(group)
group_terms <- make.names(group_levels)
group_term_map <- stats::setNames(group_terms, group_levels)
missing_covariates <- setdiff(covariates, colnames(metadata_all))
if (length(missing_covariates) > 0) {
  stop(paste("Missing covariates in metadata:", paste(missing_covariates, collapse = ", ")))
}
for (covariate in covariates) {
  metadata_all[[covariate]] <- factor(metadata_all[[covariate]])
}


requested_covariates <- covariates
usable_covariates <- covariates[vapply(
  covariates, function(cov) nlevels(metadata_all[[cov]]) >= 2, logical(1))]
dropped_covariates <- setdiff(requested_covariates, usable_covariates)
if (length(dropped_covariates) > 0) {
  message("Dropping constant covariate(s) in ", region, " / ", cell_type, ": ",
          paste(dropped_covariates, collapse = ", "))
}
covariates <- usable_covariates
design_formula <- if (length(covariates) == 0) {
  ~ 0 + group
} else {
  as.formula(paste("~ 0 + group +", paste(covariates, collapse = " + ")))
}
design <- model.matrix(design_formula, data = metadata_all)
colnames(design) <- sub("^group", "", colnames(design))
for (level in group_levels) {
  colnames(design)[colnames(design) == level] <- group_term_map[[level]]
}

y0 <- edgeR::DGEList(counts = counts_mat, group = group)
y0 <- edgeR::calcNormFactors(y0, method = "TMM")


passed_edger_filter <- edgeR::filterByExpr(y0, design = design)

summary_prefix <- file.path(out_dir, paste(region, safe_name(cell_type), sep = "__"))
plot_prefix <- file.path(plot_dir, paste(region, safe_name(cell_type), sep = "__"))

sample_qc <- data.frame(
  pseudobulk_id = rownames(metadata_all),
  metadata_all,
  edger_lib_size = y0$samples$lib.size,
  norm_factor = y0$samples$norm.factors,
  row.names = NULL,
  check.names = FALSE
)
write.csv(sample_qc, paste0(summary_prefix, "__sample_qc.csv"), row.names = FALSE)
write.csv(design, paste0(summary_prefix, "__design_matrix.csv"))
write.csv(data.frame(feature_name = rownames(y0), passed_edger_filter = passed_edger_filter), paste0(summary_prefix, "__gene_filter.csv"), row.names = FALSE)

if (nrow(y0) < 2) stop("Fewer than two genes to test.")


y <- y0
y <- edgeR::estimateDisp(y, design, robust = TRUE)
fit <- edgeR::glmQLFit(y, design, robust = TRUE)


common_disp <- y$common.dispersion
common_bcv <- sqrt(common_disp)
tagwise_bcv_median <- if (!is.null(y$tagwise.dispersion)) median(sqrt(y$tagwise.dispersion)) else NA_real_
prior_df <- if (!is.null(y$prior.df)) y$prior.df[1] else NA_real_
ql_df_prior <- if (!is.null(fit$df.prior)) fit$df.prior[1] else NA_real_

write.csv(
  data.frame(
    region = region,
    cell_type = cell_type,
    n_samples = nrow(metadata_all),
    n_features_tested = nrow(y0),
    n_features_pass_edger_filter = sum(passed_edger_filter),
    min_cells = min_cells,
    covariates = ifelse(length(covariates) == 0, "none", paste(covariates, collapse = ",")),
    covariates_requested = ifelse(length(requested_covariates) == 0, "none", paste(requested_covariates, collapse = ",")),
    covariates_dropped = ifelse(length(dropped_covariates) == 0, "none", paste(dropped_covariates, collapse = ",")),
    design_formula = paste(deparse(design_formula), collapse = ""),
    common_dispersion = common_disp,
    common_bcv = common_bcv,
    tagwise_bcv_median = tagwise_bcv_median,
    prior_df = prior_df,
    ql_df_prior = ql_df_prior,
    stringsAsFactors = FALSE
  ),
  paste0(summary_prefix, "__model_summary.csv"),
  row.names = FALSE
)

if (requireNamespace("limma", quietly = TRUE)) {
  save_static_plot(paste0(plot_prefix, "__MDS"), 1200, 1000, function() {
    limma::plotMDS(edgeR::cpm(y, log = TRUE), col = as.integer(group), labels = metadata_all$sample_id)
    legend("topright", legend = levels(group), col = seq_along(levels(group)), pch = 16, cex = 0.8)
  })
} else {
  writeLines("limma is not available; skipped MDS plot.", paste0(plot_prefix, "__MDS_skipped.txt"))
}

tryCatch({
  save_static_plot(paste0(plot_prefix, "__BCV"), 1200, 1000, function() {
    edgeR::plotBCV(y)
  })
}, error = function(e) {
  if (dev.cur() > 1) dev.off()
  writeLines(conditionMessage(e), paste0(plot_prefix, "__BCV_skipped.txt"))
})

group_counts <- table(metadata_all$group)
get_group_count <- function(tab, group_name) {
  value <- tab[group_name]
  if (is.na(value)) return(0L)
  as.integer(value)
}

for (comparison in comparisons) {
  group1 <- comparison[[1]]
  group2 <- comparison[[2]]
  comp_label <- paste(group1, "vs", group2, sep = "_")
  out_file <- file.path(out_dir, paste(region, safe_name(cell_type), comp_label, sep = "__"))
  out_file <- paste0(out_file, ".csv")

  n_group1 <- get_group_count(group_counts, group1)
  n_group2 <- get_group_count(group_counts, group2)
  if (n_group1 < 2 || n_group2 < 2) {
    skipped <- data.frame(
      feature_name = rownames(y0),
      passed_edger_filter = passed_edger_filter,
      logFC = NA_real_,
      logCPM = NA_real_,
      F = NA_real_,
      PValue = NA_real_,
      FDR = NA_real_,
      cell_type = cell_type,
      region = region,
      comparison = comp_label,
      group1 = group1,
      group2 = group2,
      n_samples_group1 = n_group1,
      n_samples_group2 = n_group2,
      n_cells_group1 = sum(metadata_all$n_cells[metadata_all$group == group1]),
      n_cells_group2 = sum(metadata_all$n_cells[metadata_all$group == group2]),
      status = "skipped_too_few_samples",
      stringsAsFactors = FALSE
    )
    write.csv(skipped, out_file, row.names = FALSE)
    next
  }

  contrast <- rep(0, ncol(design))
  names(contrast) <- colnames(design)
  term1 <- group_term_map[[group1]]
  term2 <- group_term_map[[group2]]
  if (is.null(term1) || is.null(term2) || !(term1 %in% names(contrast)) || !(term2 %in% names(contrast))) {
    stop(paste("Comparison groups are not represented in design:", group1, group2))
  }
  contrast[[term1]] <- 1
  contrast[[term2]] <- -1
  write.csv(
    data.frame(term = names(contrast), contrast = contrast, comparison = comp_label),
    paste0(summary_prefix, "__", comp_label, "__contrast.csv"),
    row.names = FALSE
  )

  qlf <- edgeR::glmQLFTest(fit, contrast = contrast)
  tab <- edgeR::topTags(qlf, n = Inf, sort.by = "none")$table
  tab$feature_name <- rownames(tab)

  all_results <- data.frame(feature_name = rownames(y0), passed_edger_filter = passed_edger_filter, stringsAsFactors = FALSE)
  all_results <- merge(all_results, tab, by = "feature_name", all.x = TRUE, sort = FALSE)
  all_results$cell_type <- cell_type
  all_results$region <- region
  all_results$comparison <- comp_label
  all_results$group1 <- group1
  all_results$group2 <- group2
  all_results$n_samples_group1 <- n_group1
  all_results$n_samples_group2 <- n_group2
  all_results$n_cells_group1 <- sum(metadata_all$n_cells[metadata_all$group == group1])
  all_results$n_cells_group2 <- sum(metadata_all$n_cells[metadata_all$group == group2])
  all_results$status <- "tested"
  write.csv(all_results, out_file, row.names = FALSE)

  
  stale_skip <- paste0(plot_prefix, "__", comp_label, "__MD_skipped.txt")
  if (file.exists(stale_skip)) file.remove(stale_skip)

  
  
  
  
  
  if (requireNamespace("limma", quietly = TRUE)) {
    tryCatch({
      de_status <- rep(0L, nrow(qlf$table))
      de_status[qlf$table$logFC > 0 & tab$FDR < 0.05] <- 1L
      de_status[qlf$table$logFC < 0 & tab$FDR < 0.05] <- -1L
      save_static_plot(paste0(plot_prefix, "__", comp_label, "__MD"), 1200, 1000, function() {
        limma::plotMD(qlf, status = de_status, values = c(1, -1),
                      col = c("#d73027", "#4575b4"), legend = "topright",
                      main = paste(cell_type, comp_label))
        abline(h = 0, col = "grey40", lty = 2)
      })
    }, error = function(e) {
      if (dev.cur() > 1) dev.off()
      writeLines(conditionMessage(e), paste0(plot_prefix, "__", comp_label, "__MD_skipped.txt"))
    })
  } else {
    writeLines("limma not available; skipped MD plot.",
               paste0(plot_prefix, "__", comp_label, "__MD_skipped.txt"))
  }

  
  
  
  
  
  
  tryCatch({
    pvals <- tab$PValue[is.finite(tab$PValue)]
    save_static_plot(paste0(plot_prefix, "__", comp_label, "__pvalue_hist"), 1100, 900, function() {
      hist(pvals, breaks = seq(0, 1, by = 0.05), col = "#9ecae1", border = "white",
           xlab = "raw p-value", main = paste("p-value distribution:", cell_type, comp_label))
      abline(h = length(pvals) / 20, col = "grey40", lty = 2)
    })
  }, error = function(e) {
    if (dev.cur() > 1) dev.off()
    writeLines(conditionMessage(e), paste0(plot_prefix, "__", comp_label, "__pvalue_hist_skipped.txt"))
  })
}
