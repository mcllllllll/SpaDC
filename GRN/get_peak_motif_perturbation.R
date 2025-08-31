library(JASPAR2024)
library(TFBSTools)
library(RSQLite)
library(motifmatchr)
library(GenomicRanges)
library(BSgenome.Mmusculus.UCSC.mm10) 
library(Biostrings)
library(stringr)
setwd("/home/nas2/biod/machuanlong/data/GRN/misar")
peak_file <- "result/all_top_peaks.bed"
output_dir <- "peak_motif_perturbation"
dir.create(output_dir, showWarnings = FALSE)

# === 1. Read peak file ===
peaks <- read.table(peak_file, header=FALSE)
colnames(peaks) <- c("chr", "start", "end")
peak_gr <- GRanges(seqnames = peaks$chr,
                   ranges = IRanges(start = peaks$start, end = peaks$end))

# === 2. Extract ±672bp around peak center (total 1344bp) ===
seq_len <- 1344
peak_mid <- (start(peak_gr) + end(peak_gr)) %/% 2
peak_start <- peak_mid - (seq_len %/% 2) + 1  
peak_end <- peak_start + seq_len - 1
peak_1344bp <- GRanges(
  seqnames = seqnames(peak_gr),
  ranges = IRanges(start = peak_start, end = peak_end)
) 
peak_ids <- paste0(seqnames(peak_1344bp), "-", start(peak_gr), "-", end(peak_gr))
names(peak_1344bp) <- peak_ids

# === 3. Load mouse motif matrices from JASPAR2024 database ===
jaspar_db <- dbConnect(RSQLite::SQLite(), "JASPAR2024.sqlite")
opts <- list(species = "Mus musculus", collection = "CORE")
out <- getMatrixSet(jaspar_db, opts)
if (!isTRUE(all.equal(name(out), names(out)))) {
  names(out) <- paste(names(out), name(out), sep = "_")
}
motifs <- out

# === 4. Match motifs on 1344bp sequences ===
motif_positions <- matchMotifs(motifs, peak_1344bp, genome = BSgenome.Mmusculus.UCSC.mm10, out = "positions")

# === 5. Combine all motif positions into one object ===
all_motifs <- do.call(c, lapply(seq_along(motif_positions), function(i) {
  m <- motif_positions[[i]]
  m$motif_id <- names(motif_positions)[i]
  return(m)
}))

# === 6. Extract original DNA sequences ===
original_seqs <- getSeq(BSgenome.Mmusculus.UCSC.mm10, peak_1344bp)

# === 7. Map peaks to motif hits ===
peak_hits <- findOverlaps(peak_1344bp, all_motifs)
motif_by_peak <- split(subjectHits(peak_hits), queryHits(peak_hits))

# === 8. Function to generate random DNA sequences ===
generate_random_seq <- function(len) {
  paste0(sample(c("A", "C", "G", "T"), len, replace = TRUE), collapse = "")
}

# === 9. Main loop to generate perturbed sequences ===
for (i in seq_along(peak_1344bp)) {
  peak_name <- peak_ids[i]
  peak_dir <- file.path(output_dir, peak_name)
  dir.create(peak_dir, showWarnings = FALSE, recursive = TRUE)

  seq_original <- as.character(original_seqs[i])
  original_fasta <- DNAStringSet(seq_original)
  names(original_fasta) <- paste0(peak_name, "_original")
  writeXStringSet(original_fasta, file.path(peak_dir, "original.fasta"))

  if (!(i %in% names(motif_by_peak))) next
  motif_indices <- motif_by_peak[[as.character(i)]]
  motifs_in_peak <- all_motifs[motif_indices]

  for (motif_id in unique(motifs_in_peak$motif_id)) {
    motif_hits <- motifs_in_peak[motifs_in_peak$motif_id == motif_id]

    motif_coords <- lapply(seq_along(motif_hits), function(k) {
      list(
        start = start(motif_hits[k]) - start(peak_1344bp[i]) + 1,
        end = end(motif_hits[k]) - start(peak_1344bp[i]) + 1
      )
    })

    # clean
    clean_id <- motif_id
    clean_id <- gsub("::", "_", clean_id)
    clean_id <- gsub("[^A-Za-z0-9_.-]", "_", clean_id)

    perturbations <- DNAStringSet()
    for (j in 1:10) {
      seq_mut <- seq_original
      for (coord in motif_coords) {
        len <- coord$end - coord$start + 1
        rand_seq <- generate_random_seq(len)
        substr(seq_mut, coord$start, coord$end) <- rand_seq
      }
      perturb_name <- paste0(peak_name, "_", clean_id, "_rand_", j)
      perturbations <- append(perturbations, DNAStringSet(seq_mut, use.names=FALSE))
      names(perturbations)[length(perturbations)] <- perturb_name
    }

    writeXStringSet(perturbations, file.path(peak_dir, paste0(clean_id, "_Perturbation.fasta")))
  }
  print(i)
}
