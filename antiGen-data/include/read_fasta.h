#pragma once

#include <vector>
#include <string>
#include <unordered_map>

struct ProcessedFasta {
    std::vector<std::vector<char>> protein_aa; // one vector per sequence (renamed from spike_aa)
    std::vector<std::string> strain_names;     // strain ID per sequence (FASTA header), parallel to protein_aa
    std::vector<std::string> dates;            // collection date per sequence (YYYY-MM-DD or "")
    std::vector<double> decimal_dates;         // decimal year (e.g. 2020.598), -inf if missing
    int n_seqs;
    int aa_seq_len;                           // dynamic protein length based on dataset
};

// Convert date string to decimal year (e.g. "2020-08-06" -> 2020.598)
double date_to_decimal(const std::string& date);

// Load metadata TSV into name -> date map.
// Expects a header row with "strain" and "date" columns.
std::unordered_map<std::string, std::string> load_date_map(const std::string& tsv_path);

// Reads protein AA alignment FASTA and looks up dates from the date map.
// max_seqs: stop after this many sequences (-1 = read all).
ProcessedFasta read_and_process_fasta(const std::string& fasta_path,
                                      const std::unordered_map<std::string, std::string>& date_map,
                                      int expected_length,
                                      int max_seqs = -1);

// Reads multiple protein segment alignment files and concatenates them.
// For H3N2: reads SigPep, HA1, HA2 files and concatenates per sequence.
ProcessedFasta read_multiple_alignments(
    const std::vector<std::string>& alignment_files,
    const std::vector<int>& segment_lengths,
    const std::unordered_map<std::string, std::string>& date_map,
    int max_seqs = -1
);

// Sort sequences by decimal date (earliest first).
void sort_by_date(ProcessedFasta& fasta);

// Flatten protein_aa into a contiguous buffer for GPU upload.
std::vector<char> flatten_protein_aa(const ProcessedFasta& fasta);
