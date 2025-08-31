library(TFBSTools)
library(RSQLite)
library(Biostrings)
setwd("/home/nas2/biod/machuanlong/data/GRN/misar")

# Connect to JASPAR2024 database
jaspar_db <- dbConnect(RSQLite::SQLite(), "JASPAR2024.sqlite")
opts <- list(species = "Mus musculus", collection = "CORE")
motifs <- getMatrixSet(jaspar_db, opts)

# read shuffled_peaks
shuffled_pks <- readDNAStringSet("Mus_musculus_motif_fasta/shuffled_peaks.fasta", format = "fasta")

dir.create("Mus_musculus_motif_fasta/shuffled_peaks_motifs/")

for (i in 1:length(motifs)) {
  tf <- paste0(motifs[[i]]@ID, "_", gsub("::", "_", motifs[[i]]@name)) 
  pfm <- Matrix(motifs[[i]])
  
  ppm <- apply(pfm, 2, function(column) {
    return(column / sum(column)) 
  })
  
  set.seed(10)
  out <- apply(ppm, 2, function(x) {
    return(sample(rownames(ppm), 1000, replace = T, prob = x))  
  })
  
  motif_seqs <- apply(out, 1, function(x) paste(x, collapse = ""))
  
  left_coord <- width(shuffled_pks)[1] / 2 - floor(ncol(ppm) / 2)
  left <- as.character(subseq(shuffled_pks, start = 1, end = left_coord))
  right <- as.character(subseq(shuffled_pks, start = left_coord + ncol(ppm) + 1, end = width(shuffled_pks)[1]))

  shuffled_pks_motifs <- DNAStringSet(paste0(left, motif_seqs, right))

  writeXStringSet(shuffled_pks_motifs, paste0("Mus_musculus_motif_fasta/shuffled_peaks_motifs/", tf, ".fasta"), format = "fasta")
  
  print(i)
}
