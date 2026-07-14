#include "read_fasta.h"
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

static bool is_leap_year(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

double date_to_decimal(const std::string& date) {
    if (date.empty()) return -INFINITY;

    // Split on '-' so components can be non-zero-padded (e.g. "2018-12-4"),
    // and any non-numeric string (e.g. "unknown") falls through to -INFINITY.
    std::vector<std::string> parts;
    std::string cur;
    for (char c : date) {
        if (c == '-') { parts.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    parts.push_back(cur);

    // First part must be a 4-digit numeric year.
    if (parts.empty() || parts[0].size() != 4) return -INFINITY;
    for (char c : parts[0]) if (!std::isdigit((unsigned char)c)) return -INFINITY;

    int year;
    try { year = std::stoi(parts[0]); } catch (...) { return -INFINITY; }
    int days_in_year = is_leap_year(year) ? 366 : 365;

    if (parts.size() == 1) {
        // Year only: middle of year
        return year + 0.5;
    }

    int month;
    try { month = std::stoi(parts[1]); } catch (...) { return -INFINITY; }
    if (month < 1 || month > 12) return -INFINITY;

    // Days at start of each month (non-leap)
    static const int month_days[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
    int day_of_year = 0;
    for (int m = 0; m < month - 1; m++) {
        day_of_year += month_days[m];
        if (m == 1 && is_leap_year(year)) day_of_year++;
    }

    if (parts.size() == 2) {
        // Year-month only: middle of month
        int mdays = month_days[month - 1];
        if (month == 2 && is_leap_year(year)) mdays++;
        return year + (day_of_year + mdays * 0.5) / days_in_year;
    }

    // Full date
    int day;
    try { day = std::stoi(parts[2]); } catch (...) { return -INFINITY; }
    if (day < 1 || day > 31) return -INFINITY;
    day_of_year += day - 1; // 0-indexed day within year (Jan 1 = 0)
    return year + (day_of_year + 0.5) / days_in_year;
}

std::unordered_map<std::string, std::string> load_date_map(const std::string& tsv_path) {
    auto t_start = std::chrono::high_resolution_clock::now();
    std::unordered_map<std::string, std::string> date_map;
    date_map.reserve(3200000);

    std::ifstream file(tsv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + tsv_path);
    }

    // Parse header to find column indices
    std::string header;
    std::getline(file, header);
    std::istringstream hss(header);
    std::string col;
    int name_col = -1, date_col = -1;
    int col_idx = 0;
    while (std::getline(hss, col, '\t')) {
        if (col == "strain") name_col = col_idx;
        else if (col == "date") date_col = col_idx;
        col_idx++;
    }
    if (name_col < 0 || date_col < 0) {
        throw std::runtime_error("TSV missing 'strain' or 'date' column");
    }

    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        // Split by tab and extract the two columns we need
        int ci = 0;
        size_t pos = 0;
        std::string name, date;
        while (pos <= line.size()) {
            size_t next = line.find('\t', pos);
            if (next == std::string::npos) next = line.size();
            if (ci == name_col) name = line.substr(pos, next - pos);
            else if (ci == date_col) date = line.substr(pos, next - pos);
            pos = next + 1;
            ci++;
            if (ci > name_col && ci > date_col) break; // no need to parse further
        }
        if (!name.empty()) {
            date_map[name] = date;
        }
        count++;
        if (count % 500000 == 0) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_start).count();
            std::cout << "  Loaded " << count << " TSV entries (" << (int)elapsed << " ms)" << std::endl;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Loaded " << count << " date entries from TSV in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
              << " ms" << std::endl;
    return date_map;
}

ProcessedFasta read_and_process_fasta(const std::string& fasta_path,
                                      const std::unordered_map<std::string, std::string>& date_map,
                                      int expected_length,
                                      int max_seqs) {
    ProcessedFasta result;
    result.n_seqs = 0;
    result.aa_seq_len = expected_length;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::ifstream file(fasta_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + fasta_path);
    }

    std::string line;

    int missing_dates = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] != '>') continue;

        // Extract sequence name: everything after '>' (trimmed)
        std::string name = line.substr(1);
        // Trim trailing whitespace/carriage return
        while (!name.empty() && (name.back() == ' ' || name.back() == '\t' || name.back() == '\r')) {
            name.pop_back();
        }

        std::string seq_line;
        if (!std::getline(file, seq_line)) break;

        if ((int)seq_line.size() < expected_length) {
            std::cerr << "Warning: sequence \"" << name << "\" has length " << seq_line.size()
                      << " (expected >= " << expected_length << "), skipping" << std::endl;
            continue;
        }

        // Store the first expected_length characters as the protein AA sequence
        std::vector<char> aa(seq_line.begin(), seq_line.begin() + expected_length);
        result.protein_aa.push_back(std::move(aa));

        // Look up date
        std::string date_str;
        auto it = date_map.find(name);
        if (it != date_map.end()) {
            date_str = it->second;
        } else {
            missing_dates++;
        }
        result.dates.push_back(date_str);
        result.decimal_dates.push_back(date_to_decimal(date_str));

        result.n_seqs++;

        if (max_seqs > 0 && result.n_seqs >= max_seqs) break;

        if (result.n_seqs % 100000 == 0) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_start).count();
            std::cout << "  Processed " << result.n_seqs << " sequences (" << (int)elapsed << " ms)" << std::endl;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Processed " << result.n_seqs << " sequences in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
              << " ms" << std::endl;
    if (missing_dates > 0) {
        std::cerr << "Warning: " << missing_dates << " sequences had no date in metadata" << std::endl;
    }
    return result;
}

ProcessedFasta read_multiple_alignments(
    const std::vector<std::string>& alignment_files,
    const std::vector<int>& segment_lengths,
    const std::unordered_map<std::string, std::string>& date_map,
    int max_seqs
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    ProcessedFasta result;
    result.n_seqs = 0;
    result.aa_seq_len = 0;
    for (int len : segment_lengths) result.aa_seq_len += len;

    std::cout << "Reading " << alignment_files.size() << " alignment files..." << std::endl;

    // Read all segment files and store sequences per strain
    std::unordered_map<std::string, std::vector<std::string>> strain_segments;
    std::vector<std::string> strain_order; // Track order from first file

    for (size_t file_idx = 0; file_idx < alignment_files.size(); file_idx++) {
        std::ifstream file(alignment_files[file_idx]);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open " + alignment_files[file_idx]);
        }

        std::string line;
        int sequences_read = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] != '>') continue;

            std::string strain = line.substr(1);
            // Trim trailing whitespace
            while (!strain.empty() && (strain.back() == ' ' || strain.back() == '\t' || strain.back() == '\r')) {
                strain.pop_back();
            }

            std::string seq_line;
            if (!std::getline(file, seq_line)) break;

            if ((int)seq_line.size() < segment_lengths[file_idx]) {
                std::cerr << "Warning: sequence \"" << strain << "\" in file " << file_idx
                          << " has length " << seq_line.size()
                          << " (expected >= " << segment_lengths[file_idx] << "), skipping" << std::endl;
                continue;
            }

            // Store segment for this strain
            if (strain_segments.find(strain) == strain_segments.end()) {
                strain_segments[strain] = std::vector<std::string>(alignment_files.size());
                if (file_idx == 0) strain_order.push_back(strain); // Track order from first file
            }
            strain_segments[strain][file_idx] = seq_line.substr(0, segment_lengths[file_idx]);

            sequences_read++;
            if (sequences_read % 50000 == 0) {
                std::cout << "  File " << file_idx << ": " << sequences_read << " sequences" << std::endl;
            }
        }

        std::cout << "File " << file_idx << " (" << alignment_files[file_idx] << "): "
                  << sequences_read << " sequences" << std::endl;
    }

    // Build concatenated sequences for strains present in all files
    int missing_dates = 0;
    for (const std::string& strain : strain_order) {
        auto& segments = strain_segments[strain];

        // Check if all segments are present
        bool all_present = true;
        for (size_t i = 0; i < segments.size(); i++) {
            if (segments[i].empty()) {
                all_present = false;
                break;
            }
        }

        if (!all_present) continue;

        // Concatenate segments
        std::string concatenated;
        for (const std::string& segment : segments) {
            concatenated += segment;
        }

        result.protein_aa.emplace_back(concatenated.begin(), concatenated.end());

        // Look up date
        std::string date_str;
        auto it = date_map.find(strain);
        if (it != date_map.end()) {
            date_str = it->second;
        } else {
            missing_dates++;
        }
        result.dates.push_back(date_str);
        result.decimal_dates.push_back(date_to_decimal(date_str));

        result.n_seqs++;

        if (max_seqs > 0 && result.n_seqs >= max_seqs) break;

        if (result.n_seqs % 100000 == 0) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_start).count();
            std::cout << "  Concatenated " << result.n_seqs << " sequences (" << (int)elapsed << " ms)" << std::endl;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Concatenated " << result.n_seqs << " sequences in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
              << " ms" << std::endl;
    if (missing_dates > 0) {
        std::cerr << "Warning: " << missing_dates << " sequences had no date in metadata" << std::endl;
    }

    return result;
}

void sort_by_date(ProcessedFasta& fasta) {
    std::vector<int> order(fasta.n_seqs);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return fasta.decimal_dates[a] < fasta.decimal_dates[b];
    });

    std::vector<std::vector<char>> sorted_aa(fasta.n_seqs);
    std::vector<std::string> sorted_dates(fasta.n_seqs);
    std::vector<double> sorted_decimal(fasta.n_seqs);
    for (int i = 0; i < fasta.n_seqs; i++) {
        sorted_aa[i] = std::move(fasta.protein_aa[order[i]]);
        sorted_dates[i] = std::move(fasta.dates[order[i]]);
        sorted_decimal[i] = fasta.decimal_dates[order[i]];
    }
    fasta.protein_aa = std::move(sorted_aa);
    fasta.dates = std::move(sorted_dates);
    fasta.decimal_dates = std::move(sorted_decimal);
}

std::vector<char> flatten_protein_aa(const ProcessedFasta& fasta) {
    std::vector<char> flat((long long)fasta.n_seqs * fasta.aa_seq_len);
    for (int i = 0; i < fasta.n_seqs; i++) {
        memcpy(flat.data() + (long long)i * fasta.aa_seq_len, fasta.protein_aa[i].data(), fasta.aa_seq_len);
    }
    return flat;
}
