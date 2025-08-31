library(Biostrings)
library(rtracklayer)
library(BSgenome.Mmusculus.UCSC.mm10) 
setwd("/home/nas2/biod/machuanlong/data/GRN/misar")

get.seqs <- function (org, regions, no.cores=1) {

  ## Function for chromosome
  get.seq.chr <- function (chr) {
    seq <- org[[chr]]
    if (class (seq) == 'MaskedDNAString')
      seq <- unmasked (seq)

    ## Get all sequences
    chr.regions <- regions[which (as.character (seqnames (regions)) == chr)]
    region.seq <- DNAStringSet (Views (seq, start=start (chr.regions), end=end (chr.regions)))
    ## Reverse complement negative strands
    rev <- as.logical (strand (chr.regions) == '-')
    region.seq[rev] <- reverseComplement (region.seq[rev])
    names (region.seq) <- values (chr.regions)$.ZZ
    gc ()
    return (region.seq)
  }

  ## Split by chromosomes
  values (regions)$.ZZ <- sprintf ("Unit%d", 1:length (regions))
  seqs <- unique (as.character (seqnames (regions)))

  ## Run in parallel
  all.seqs <- lapply (seqs, get.seq.chr)
  all.seqs <- do.call (c, all.seqs)

  ## Sort to original order
  inds <- sort (as.numeric (gsub ('Unit', '', names (all.seqs))), index.return=TRUE)$ix
  all.seqs <- all.seqs[inds]

  # Clean up
  all.seqs <- as.character (all.seqs)
  gc ()
  all.seqs <- DNAStringSet (all.seqs)
  gc ()
  
  return (all.seqs)
}

# we used command line script fasta_ushuffle to generate shuffled sequences, get shuffled_peaks.fasta
dir.create("Mus_musculus_motif_fasta")

# Origin: Generated from the Peak-Gene section in Tutorials/GRN.ipynb
bed <- import.bed("result/all_top_peaks.bed")  
bed <- resize(bed, width = 1344, fix = "center")
set.seed(10)
examples <- sample(bed, 1000)
seqs <- get.seqs(BSgenome.Mmusculus.UCSC.mm10, examples, 1)
writeXStringSet(seqs, "Mus_musculus_motif_fasta/example_peaks.fasta", format = "fasta", width = 1344)

cmd1 <- "sed 's/\r$//' data/GRN/misar/Mus_musculus_motif_fasta/example_peaks.fasta > data/GRN/misar/Mus_musculus_motif_fasta/example_peaks_clean.fasta"
system(cmd1)
cmd2 <- "fasta_ushuffle/fasta_ushuffle -k 2 < data/GRN/misar/Mus_musculus_motif_fasta/example_peaks_clean.fasta > data/GRN/misar/Mus_musculus_motif_fasta/shuffled_peaks.fasta"
system(cmd2)


  
