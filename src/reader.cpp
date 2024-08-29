#include <algorithm>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "array_support.h"
#include "vtk_support.h"

using namespace nb::literals;

/* We are on Windows */
#if defined(_WIN32) || defined(_WIN64)
#define strtok_r strtok_s
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// #define DEBUG

static const double DIV_OF_TEN[] = {
    1.0e-0,  1.0e-1,  1.0e-2,  1.0e-3,  1.0e-4,  1.0e-5,  1.0e-6,  1.0e-7,  1.0e-8,  1.0e-9,
    1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13, 1.0e-14, 1.0e-15, 1.0e-16, 1.0e-17, 1.0e-18, 1.0e-19,
    1.0e-20, 1.0e-21, 1.0e-22, 1.0e-23, 1.0e-24, 1.0e-25, 1.0e-26, 1.0e-27,
};

static inline double power_of_ten(int exponent) {
    double result = 1.0;
    double base = (exponent < 0) ? 0.1 : 10.0;
    int abs_exponent = abs(exponent);
    for (int i = 0; i < abs_exponent; ++i) {
        result *= base;
    }
    return result;
}

// Fast string to integer converter
static inline int fast_atoi(const char *raw, const int intsz) {
    int val = 0;
    int c;
    char current_char;

    for (c = 0; c < intsz; ++c) {
        current_char = *raw++;

        // Multiply by 10 only if current_char is a digit
        if (current_char >= '0' && current_char <= '9') {
            val = val * 10 + (current_char - '0');
        }
    }

    return val;
}

// Checks for negative character
static inline int checkneg(char *raw, int intsz) {
    int c;
    for (c = 0; c < intsz; ++c) {
        if (raw[0] == '-') {
            return 1;
        }
        ++raw;
    }
    return 0;
}

// Reads fortran formatted float formats in the form of
// "3.7826539829200E+00"
// "1.0000000000000E-001"
//
// fltsz : Number of characters to read in a floating point number
static inline int strtod(char *raw, int fltsz, double *arr) {
    char *end = raw + fltsz;
    double sign = 1;

    while (raw < end) {
        if (*raw == '\0') {
            // section empty, value is zero
            *arr = 0;
            return 1;
        } else if (*raw != ' ') { // always skip whitespace
            break;
        }
        raw++;
    }

    // either a number or a sign
    if (*raw == '-') {
        sign = -1;
        ++raw;
    }

    // next value is always a number
    // Use integer arithmetric and then convert to a float
    uint64_t val_int = *raw++ - '0';
    raw++; // next value is always a "."

    // Read through the rest of the number
    int decimal_digits = 0;
    while (raw < end) {
        if (*raw == 'e' || *raw == 'E') { // incredibly, can be lowercase
            break;
        } else if (*raw >= '0' && *raw <= '9') {
            val_int = val_int * 10 + (*raw++ - '0');
            decimal_digits++;
        }
    }

    // Compute the floating-point value
    double val;
    if (decimal_digits < 27) {
        val = (double)val_int * DIV_OF_TEN[decimal_digits];
    } else {
        val = (double)val_int * power_of_ten(10);
    }

    // Might have scientific notation remaining, for example:
    // 1.0000000000000E-001
    int evalue = 0;
    int esign = 1;
    if (*raw == 'e' || *raw == 'E') {
        raw++; // skip "E"
        // always a sign of some sort
        if (*raw == '-') {
            esign = -1;
        }
        raw++;

        while (raw < end) {
            // read to whitespace or end of the line
            if (*raw == ' ' || *raw == '\0') {
                break;
            }
            evalue = evalue * 10 + (*raw++ - '0');
        }
        if (esign == 1) {
            val *= power_of_ten(evalue);
        } else {
            val /= power_of_ten(evalue);
        }
    }

    // seek through end of float value
    if (sign == -1) {
        *arr = -val;
    } else {
        *arr = val;
    }

    // check if at end of line
    return 0;
}

class MemoryMappedFile {
  private:
    size_t size;
    char *start;
#ifdef _WIN32
    HANDLE fileHandle;
    HANDLE mapHandle;
#else
    int fd;
#endif

  public:
    std::string line;
    char *current;

    MemoryMappedFile(const char *filename)
        : start(nullptr), current(nullptr), size(0)
#ifdef _WIN32
          ,
          fileHandle(INVALID_HANDLE_VALUE), mapHandle(nullptr)
#else
          ,
          fd(-1)
#endif
    {
#ifdef _WIN32
        fileHandle = CreateFile(
            filename,
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (fileHandle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Error opening file");
        }

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(fileHandle, &fileSize)) {
            CloseHandle(fileHandle);
            throw std::runtime_error("Error getting file size");
        }

        size = static_cast<size_t>(fileSize.QuadPart);
        mapHandle = CreateFileMapping(fileHandle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapHandle == nullptr) {
            CloseHandle(fileHandle);
            throw std::runtime_error("Error creating file mapping");
        }

        start = static_cast<char *>(MapViewOfFile(mapHandle, FILE_MAP_READ, 0, 0, size));
        if (start == nullptr) {
            CloseHandle(mapHandle);
            CloseHandle(fileHandle);
            throw std::runtime_error("Error mapping file");
        }
#else
        fd = open(filename, O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Error opening file");
        }

        struct stat st;
        if (fstat(fd, &st) == -1) {
            close(fd);
            throw std::runtime_error("Error getting file size");
        }

        size = st.st_size;
        start = static_cast<char *>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (start == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Error mapping file");
        }
#endif
        current = start;
    }

    ~MemoryMappedFile() {
        close_file();
#ifdef _WIN32
        if (fileHandle != INVALID_HANDLE_VALUE) {
            CloseHandle(fileHandle);
        }
        if (mapHandle != nullptr) {
            CloseHandle(mapHandle);
        }
#else
        if (fd != -1) {
            close(fd);
        }
#endif
    }

    void close_file() {
        if (start) {
#ifdef _WIN32
            UnmapViewOfFile(start);
#else
            munmap(start, size);
#endif
            start = nullptr;
            current = nullptr;
        }
    }

    char &operator[](size_t index) {
        // implement bounds checking?
        // if (index >= size) {
        //     throw std::out_of_range("Index out of bounds");
        // }
        return current[index];
    }

    void operator+=(size_t offset) { current += offset; }

    // Seek to the end of the line
    void seek_eol() {
        // check if at end of file
        if (current >= start + size) {
            // std::cout << "end" << std::endl;
            return;
        }

        while (current < start + size && *current != '\n') {
            current++;
        }

        if (current < start + size && *current == '\n') {
            current++;
        }
    }

    bool eof() { return current >= start + size; }

    bool read_line() {
        line.clear();
        if (current >= start + size) {
            return false;
        }

        char *line_start = current;
        while (current < start + size && *current != '\n') {
            line += *current++;
        }

        if (current < start + size && *current == '\n') {
            current++;
        }

        return line_start != current;
    }

    size_t current_line_length() const {
        char *temp = current;
        size_t length = 0;
        while (temp < start + size && *temp != '\n') {
            length++;
            temp++;
        }
        return length;
    }

    off_t tellg() const { return current - start; }
};


int ReadNBlockMemMap(
    MemoryMappedFile &memmap,
    int *nnum,
    double *nodes,
    double *node_angles,
    const int nnodes,
    std::array<int, 3> d_size,
    const int f_size) {

    int i, j, i_val, eol;
    for (i = 0; i < nnodes; i++) {
        // Read a line from the file
        int count = memmap.current_line_length();

        // It's possible that less nodes are written to the record than
        // indicated.  In this case the line starts with a -1
        if (memmap[0] == '-') {
            break;
        }

        i_val = fast_atoi(memmap.current, d_size[0]);
#ifdef DEBUG
        std::cout << "Node number " << i_val << std::endl;
#endif
        nnum[i] = i_val;

        memmap += d_size[0];
        memmap += d_size[1];
        memmap += d_size[2];

        // read nodes
        int n_read = (count - d_size[0] * 3) / f_size;
#ifdef DEBUG
        std::cout << "n_read: " << n_read << std::endl;
#endif
        int n_read_nodes = (n_read < 3) ? n_read : 3;
        for (j = 0; j < n_read_nodes; j++) {
            ans_strtod(memmap.current, f_size, &nodes[3 * i + j]);
#ifdef DEBUG
            std::cout << " " << nodes[3 * i + j] << " ";
#endif
            memmap.current += f_size;
        }
#ifdef DEBUG
        std::cout << std::endl;
#endif

        // fill in unread nodes with zeros
        for (; j < 3; j++) {
            nodes[3 * i + j] = 0;
        }

        // read in node angles if applicable
        for (; j < n_read; j++) {
            eol = ans_strtod(memmap.current, f_size, &node_angles[3 * i + (j - 3)]);
            if (eol)
                break;
            memmap += f_size;
        }

        // seek to the end of the line
        memmap.seek_eol();
    }

    return i;
}

class Deck {

  private:
    // std::string line;
    bool debug;
    std::string filename;
    MemoryMappedFile memmap;

  public:
    bool read_parameters;

    bool read_eblock;
    bool eblock_is_read = false;
    bool nblock_is_read = false;

    std::vector<std::vector<int>> elem_type;
    std::unordered_map<int, std::vector<std::vector<int>>> keyopt;

    // Element block
    int n_elem = 0;
    NDArray<int, 1> elem_arr;
    NDArray<int, 1> elem_off_arr;

    // Node block
    int n_nodes = 0;
    int nblock_start = 0;
    int nblock_end = 0;
    NDArray<double, 2> nodes_arr;
    NDArray<double, 2> node_angles_arr;
    NDArray<int, 1> nnum_arr;

    // real constants
    std::vector<int> rnum;                 // real constant number
    std::vector<std::vector<double>> rdat; // real constant data

    // components
    std::unordered_map<std::string, NDArray<int, 1>> elem_comps;
    std::unordered_map<std::string, NDArray<int, 1>> node_comps;

    Deck(
        const std::string &fname,
        bool readParams = false,
        bool dbg = false,
        bool readEblock = true)
        : filename(fname), read_elements(readEblock), debug(dbg)
          memmap(fname.c_str()) {

        // Likely bogus leak warnings. See:
        // https://nanobind.readthedocs.io/en/latest/faq.html#why-am-i-getting-errors-about-leaked-functions-and-types
        nb::set_leak_warnings(false);

    }

    // // EXAMPLE: ET, 4, 186
    // // Element type, element number, element type
    // void ReadETLine() {
    //     if (debug) {
    //         std::cout << "reading ET" << std::endl;
    //     }

    //     // Assumes that the memory map is already at the ET line
    //     std::istringstream iss(memmap.line);
    //     std::string token;
    //     std::vector<int> et_vals;

    //     // Must start with ET (simply skip first token)
    //     std::getline(iss, token, ',');
    //     // if (token.compare(0, 2, "ET") != 0 && token.compare(0, 2, "et") != 0) {
    //     //     return;
    //     // }

    //     while (std::getline(iss, token, ',')) {
    //         try {
    //             et_vals.push_back(std::stoi(token));
    //         } catch (const std::invalid_argument &) {
    //             return;
    //         }
    //     }
    //     if (et_vals.size() == 2) {
    //         elem_type.push_back(et_vals);
    //     }
    // }

    // // Read ETBLOCK
    // void ReadETBlock() {
    //     std::string token;
    //     std::vector<int> values;

    //     // Assumes current line is an ETBLOCK
    //     std::istringstream iss(memmap.line);

    //     // skip first item (ETBLOCK)
    //     std::getline(iss, token, ',');

    //     // read the number of items in the block
    //     std::getline(iss, token, ',');
    //     int n_items = 0;
    //     try {
    //         n_items = std::stoi(token);
    //     } catch (...) {
    //         std::cerr << "Failed to read ETBLOCK n_items" << std::endl;
    //         return;
    //     }

    //     // Skip format line (2i9,19a9)
    //     memmap.seek_eol();

    //     // Read each item
    //     int elem_id, elem_type_id;
    //     for (int i = 0; i < n_items; i++) {
    //         // std::getline(cfile, line);
    //         memmap.read_line();
    //         iss.clear();
    //         iss.str(memmap.line);

    //         // We only care about the first two integers in the block, the rest are keyopts
    //         if (!(iss >> elem_id >> elem_type_id)) {
    //             // allow a soft return
    //             std::cerr << "Failed to read ETBLOCK entry at line: " << memmap.line
    //                       << std::endl;
    //             return;
    //         }
    //         std::vector<int> entry;
    //         entry.push_back(elem_id);
    //         entry.push_back(elem_type_id);
    //         elem_type.push_back(entry);
    //     }
    // }

    // // Read EBLOCK
    // void ReadEBlock() {
    //     // Sometimes, DAT files contain two EBLOCKs. Read only the first block.
    //     if (eblock_is_read) {
    //         return;
    //     }

    //     // Assumes already start of EBLOCK
    //     std::istringstream iss(memmap.line);

    //     // Only read in SOLID eblocks
    //     std::transform(
    //         memmap.line.begin(), memmap.line.end(), memmap.line.begin(), [](unsigned char c) {
    //             return std::tolower(c);
    //         });
    //     if (memmap.line.find("solid") == std::string::npos) {
    //         return;
    //     }

    //     // Get size of EBLOCK from the last item in the line
    //     // Example: "EBLOCK,19,SOLID,,3588"
    //     n_elem = getSizeOfEBLOCK(memmap.line);

    //     // we have to allocate memory for the maximum size since we don't know that a priori
    //     int *elem_data = AllocateArray<int>(n_elem * 30);
    //     elem_off_arr = MakeNDArray<int, 1>({n_elem + 1});
    //     int elem_sz = ReadEBlockMemMap(memmap, elem_off_arr.data(), elem_data, n_elem);

    //     // wrap the raw data but limit it to the size of the number of elements read
    //     // Note: This is faster than reallocating a new array but will use more memory
    //     elem_arr = WrapNDarray<int, 1>(elem_data, {elem_sz});

    //     // must mark eblock is read since we can only read one eblock per archive file
    //     eblock_is_read = true;
    // }

    // // Read:
    // // KEYOPT, ITYPE, KNUM, VALUE
    // void ReadKEYOPTLine() {
    //     if (debug) {
    //         std::cout << "reading KEYOPT" << std::endl;
    //     }

    //     // Assumes at KEYOPT line
    //     std::istringstream iss(memmap.line);
    //     std::string token;

    //     // Skip the first token (KEYOPT)
    //     std::getline(iss, token, ',');

    //     // element number is the first item
    //     int etype;
    //     std::getline(iss, token, ',');
    //     try {
    //         etype = std::stoi(token);
    //     } catch (const std::invalid_argument &) {
    //         // soft error
    //         std::cerr << "Invalid format in KEYOPT line for elem_num:" << memmap.line
    //                   << std::endl;
    //         return;
    //     }

    //     // Read the rest of the values (Element Type, Key Option, Value)
    //     std::vector<int> keyopt_vals;
    //     while (std::getline(iss, token, ',')) {
    //         try {
    //             keyopt_vals.push_back(std::stoi(token));
    //         } catch (const std::invalid_argument &) {
    //             // soft error
    //             std::cerr << "Invalid format in KEYOPT line:" << memmap.line << std::endl;
    //             return;
    //         }
    //     }

    //     if (keyopt_vals.size() == 2) {
    //         keyopt[etype].push_back(keyopt_vals);
    //     }
    // }

    // // Read RLBLOCK
    // void ReadRLBLOCK() {
    //     if (debug) {
    //         std::cout << "reading RLBLOCK" << std::endl;
    //     }

    //     std::vector<int> set_dat;

    //     // Assumes line at RLBLOCK
    //     std::istringstream iss(memmap.line);
    //     std::string token;
    //     // while (std::getline(iss, token, ',')) {
    //     //     set_dat.push_back(std::stoi(token));
    //     // }

    //     // skip starting RLBLOCK command
    //     std::getline(iss, token, ',');

    //     int nset, maxset, maxitems, nperline;
    //     try {
    //         std::getline(iss, token, ',');
    //         nset = std::stoi(token);
    //     } catch (...) {
    //         std::cerr << "Error RLBLOCK line when reading nset:" << memmap.line << std::endl;
    //         return;
    //     }
    //     try {
    //         std::getline(iss, token, ',');
    //         maxset = std::stoi(token);
    //     } catch (...) {
    //         std::cerr << "Error RLBLOCK line when reading maxset:" << memmap.line
    //                   << std::endl;
    //         return;
    //     }
    //     try {
    //         std::getline(iss, token, ',');
    //         maxitems = std::stoi(token);
    //     } catch (...) {
    //         std::cerr << "Error RLBLOCK line when reading maxitems:" << memmap.line
    //                   << std::endl;
    //         return;
    //     }
    //     try {
    //         std::getline(iss, token, ',');
    //         nperline = std::stoi(token);
    //     } catch (...) {
    //         std::cerr << "Error RLBLOCK line when reading nperline:" << memmap.line
    //                   << std::endl;
    //         return;
    //     }

    //     // Skip next two lines
    //     memmap.seek_eol(); // (2i8,6g16.9)
    //     memmap.seek_eol(); // (7g16.9)

    //     for (int i = 0; i < nset; i++) {
    //         // read real constant data in the form of:
    //         //        2       6  1.00000000     0.566900000E-07  0.00000000      0.000...
    //         memmap.read_line();
    //         std::vector<double> rcon;

    //         // real constant number
    //         try {
    //             rnum.push_back(std::stoi(memmap.line.substr(0, 8)));
    //         } catch (...) {
    //             std::cerr << "Failed to read real constant ID when reading:" << memmap.line
    //                       << std::endl;
    //             return;
    //         }

    //         int ncon = std::stoi(memmap.line.substr(8, 8));
    //         if (ncon > 6) {
    //             for (int j = 0; j < 6; j++) {
    //                 rcon.push_back(std::stod(memmap.line.substr(16 + 16 * j, 16)));
    //                 ncon--;
    //             }

    //             memmap.read_line();

    //             while (ncon > 0) {
    //                 for (int j = 0; j < 7 && ncon > 0; j++) {
    //                     rcon.push_back(std::stod(memmap.line.substr(16 * j, 16)));
    //                     ncon--;
    //                 }
    //                 if (ncon > 0) {
    //                     memmap.read_line();
    //                 }
    //             }
    //         } else {
    //             for (int j = 0; j < ncon; j++) {
    //                 rcon.push_back(std::stod(memmap.line.substr(16 + 16 * j, 16)));
    //             }
    //         }

    //         rdat.push_back(rcon);
    //     }
    // }

    // void ReadNBlock(const int pos) {

    //     // Sometimes, DAT files contains multiple node blocks. Read only the first block.
    //     if (nblock_is_read) {
    //         return;
    //     }
    //     nblock_start = pos;

    //     // Get size of NBLOCK
    //     // Assumes line is at NBLOCK
    //     // std::cout << "line: " << memmap.line << std::endl;
    //     try {
    //         // Number of nodes is last item in string
    //         n_nodes = std::stoi(memmap.line.substr(memmap.line.rfind(',') + 1));
    //     } catch (...) {
    //         std::cerr << "Failed to read number of nodes when reading:" << memmap.line
    //                   << std::endl;
    //         return;
    //     }
    //     if (debug) {
    //         std::cout << "Reading " << n_nodes << " nodes" << std::endl;
    //     }

    //     int *nnum_data = AllocateArray<int>(n_nodes);
    //     double *nodes_data = AllocateArray<double>(n_nodes * 3);

    //     // often angles aren't written, so it makes sense to initialize this to
    //     // zero
    //     double *node_angles_data = AllocateArray<double>(n_nodes * 3, true);

    //     // Get format of nblock
    //     memmap.read_line();
    //     NodeBlockFormat nfmt = GetNodeBlockFormat(memmap.line);

    //     // Return actual number of nodes read and wrap the raw data
    //     // std::cout << memmap.tellg() << std::endl;
    //     n_nodes = ReadNBlockMemMap(
    //         memmap,
    //         nnum_data,
    //         nodes_data,
    //         node_angles_data,
    //         n_nodes,
    //         nfmt.d_size,
    //         nfmt.f_size);
    //     nnum_arr = WrapNDarray<int, 1>(nnum_data, {n_nodes});
    //     nodes_arr = WrapNDarray<double, 2>(nodes_data, {n_nodes, 3});
    //     node_angles_arr = WrapNDarray<double, 2>(node_angles_data, {n_nodes, 3});
    //     // std::cout << memmap.tellg() << std::endl;

    //     // Read final line, this is always "N,R5.3,LOC, -1," and store file
    //     // position. This is used for later access (or rewrite) of the node
    //     // block.
    //     memmap.seek_eol();
    //     nblock_end = memmap.tellg();

    //     // Must mark nblock is read since we can only read one nblock per archive file.
    //     nblock_is_read = true;
    //     if (debug) {
    //         // std::cout << "Last line is: " << memmap.line << std::endl;
    //         std::cout << "NBLOCK complete, read " << n_nodes << " nodes." << std::endl;
    //     }
    // }

    // // Read CMBLOCK
    // void ReadCMBlock() {
    //     if (debug) {
    //         std::cout << "reading CMBLOCK" << std::endl;
    //     }

    //     // Assumes line at CMBLOCK
    //     std::istringstream iss(memmap.line);
    //     std::string token;
    //     std::vector<std::string> split_line;

    //     while (std::getline(iss, token, ',')) {
    //         split_line.push_back(token);
    //     }

    //     if (split_line.size() < 4) {
    //         std::cerr << "Poor formatting of CMBLOCK: " << memmap.line << std::endl;
    //         return;
    //     }

    //     int ncomp;
    //     try {
    //         ncomp = std::stoi(split_line[3].substr(0, split_line[3].find('!')).c_str());
    //     } catch (...) {
    //         std::cerr << "Poor formatting of CMBLOCK: " << memmap.line << std::endl;
    //         return;
    //     }

    //     memmap.read_line();
    //     int isz;
    //     try {
    //         isz = std::stoi(memmap.line.substr(
    //             memmap.line.find('i') + 1,
    //             memmap.line.find(')') - memmap.line.find('i') - 1));
    //     } catch (...) {
    //         std::cerr << "Failed to read integer size in CMBLOCK" << std::endl;
    //         return;
    //     }

    //     int nblock;
    //     try {
    //         nblock = std::stoi(memmap.line.substr(
    //             memmap.line.find('(') + 1,
    //             memmap.line.find('i') - memmap.line.find('(') - 1));
    //     } catch (...) {
    //         std::cerr << "Failed to read number of integers per line in CMBLOCK" << std::endl;
    //         return;
    //     }

    //     std::vector<int> component(ncomp);
    //     std::string tempstr(isz, '\0');

    //     for (int i = 0; i < ncomp; ++i) {
    //         if (i % nblock == 0) {
    //             memmap.read_line();
    //         }
    //         try {
    //             component[i] = std::stoi(memmap.line.substr(isz * (i % nblock), isz));
    //         } catch (...) {
    //             std::cerr << "Failed to parse integer from CMBLOCK line: " << memmap.line
    //                       << std::endl;
    //             return;
    //         }
    //     }

    //     // get component name
    //     std::string comname = split_line[1];
    //     comname.erase(comname.find_last_not_of(' ') + 1, std::string::npos);

    //     std::string line_comp_type = split_line[2];

    //     if (line_comp_type.find("NODE") != std::string::npos) {
    //         node_comps[comname] = InterpretComponent(component);
    //     } else if (line_comp_type.find("ELEM") != std::string::npos) {
    //         elem_comps[comname] = InterpretComponent(component);
    //     }

    //     if (debug) {
    //         std::cout << "Done reading CMBLOCK" << std::endl;
    //     }
    // }

    // Read the LS-DYNA deck by reading the start of each line and checking if
    // it matches a keyword. For example, *NODE denotes a node card.
    // Note: Cards appear to start with the * char.
    void Read() {
        int first_char, next_char;

        int position_start, position_end;
        while (true) {
            // Parse based on the first character rather than reading the entire
            // line. It's faster and the parsing logic is always based on the first
            // character

            if (memmap.eof()) {
#ifdef DEBUG
                std::cout << "Reached EOF" << std::endl;
#endif
                break;
            }

            first_char = memmap[0];
#ifdef DEBUG
            std::cout << "Read character: " << static_cast<char>(first_char) << std::endl;
#endif
            // Check if starting a card
            if (first_char == '*') {
                memmap.read_line();

                if (debug) {
                    std::cout << "Read *" << std::endl;
                }

                // Record element type
                if (memmap.line.compare(0, 5, "*NODE") == 0) {
                    ReadNodeCard();
                }
            } else {
                if (debug) {
                    std::cout << "No match, continuing..." << std::endl;
                }
                // Skip remainder of the line
                memmap.seek_eol();
            }
        }
    }

    // Read line and return the position prior to reading the line
    int ReadLine() { return memmap.read_line(); }

    // // Convert ansys style connectivity to VTK connectivity
    // // type_ref is a mapping between ansys element types and VTK element types
    // nb::tuple ToVTK(NDArray<int, 1> type_ref) {
    //     NDArray<int64_t, 1> offset_arr = MakeNDArray<int64_t, 1>({n_elem + 1});
    //     NDArray<uint8_t, 1> celltypes_arr = MakeNDArray<uint8_t, 1>({n_elem});

    //     // Allocate connectivity: max cell size 20 for VTK_HEXAHEDRAL
    //     int64_t *cell_data = new int64_t[n_elem * 20];

    //     int loc = ans_to_vtk(
    //         n_elem,
    //         elem_arr.data(),
    //         elem_off_arr.data(),
    //         type_ref.data(),
    //         n_nodes,
    //         nnum_arr.data(),
    //         offset_arr.data(),
    //         cell_data,
    //         celltypes_arr.data());

    //     NDArray<int64_t, 1> cells_arr = WrapNDarray<int64_t, 1>(cell_data, {loc});
    //     return nb::make_tuple(offset_arr, celltypes_arr, cells_arr);
    // }

    ~Deck() { memmap.close_file(); }

}; // Archive

// Convert ansys style connectivity to VTK connectivity
// type_ref is a mapping between ansys element types and VTK element types
// def ans_vtk_convert(
//     elem: NDArray[np.int32],
//     elem_off: NDArray[np.int32],
//     type_ref: NDArray[np.int32],
//     nnum: NDArray[np.int32],
// ) -> Tuple[NDArray[U], NDArray[np.uint8], NDArray[U]]: ...

nb::tuple ConvertToVTK(
    NDArray<int, 1> elem_arr,
    NDArray<int, 1> elem_off_arr,
    NDArray<int, 1> type_ref,
    NDArray<int, 1> nnum_arr) {

    int n_elem = elem_off_arr.size() - 1;
    int n_nodes = nnum_arr.size();

    NDArray<int64_t, 1> offset_arr = MakeNDArray<int64_t, 1>({n_elem + 1});
    NDArray<uint8_t, 1> celltypes_arr = MakeNDArray<uint8_t, 1>({n_elem});

    // Allocate connectivity: max cell size 20 for VTK_HEXAHEDRAL
    int64_t *cell_data = new int64_t[n_elem * 20];

    int loc = ans_to_vtk(
        n_elem,
        elem_arr.data(),
        elem_off_arr.data(),
        type_ref.data(),
        n_nodes,
        nnum_arr.data(),
        offset_arr.data(),
        cell_data,
        celltypes_arr.data());

    NDArray<int64_t, 1> cells_arr = WrapNDarray<int64_t, 1>(cell_data, {loc});
    return nb::make_tuple(offset_arr, celltypes_arr, cells_arr);
}

NB_MODULE(_reader, m) {
    m.def("ans_to_vtk", &ConvertToVTK);
    nb::class_<Deck>(m, "Deck")
        .def(
            nb::init<const std::string &, bool, bool, bool>(),
            "fname"_a,
            "read_params"_a = false,
            "debug"_a = false,
            "read_eblock"_a = true,
            "This class represents a LS-DYNA keyword deck.")
        .def_ro("n_elem", &Archive::n_elem)
        .def_ro("elem_type", &Archive::elem_type)
        .def_ro("elem", &Archive::elem_arr, nb::rv_policy::automatic)
        .def_ro("elem_off", &Archive::elem_off_arr, nb::rv_policy::automatic)
        .def_ro("keyopt", &Archive::keyopt)
        .def_ro("rdat", &Archive::rdat)
        .def_ro("rnum", &Archive::rnum)
        .def_ro("elem_comps", &Archive::elem_comps, nb::rv_policy::automatic)
        .def_ro("node_comps", &Archive::node_comps, nb::rv_policy::automatic)
        .def_ro("nodes", &Archive::nodes_arr, nb::rv_policy::automatic)
        .def_ro("node_angles", &Archive::node_angles_arr, nb::rv_policy::automatic)
        .def_ro("nnum", &Archive::nnum_arr, nb::rv_policy::automatic)
        .def_ro("n_nodes", &Archive::n_nodes)
        .def_ro("nblock_start", &Archive::nblock_start)
        .def_ro("nblock_end", &Archive::nblock_end)
        .def("to_vtk", &Archive::ToVTK)
        .def("read", &Archive::Read)
        .def("read_line", &Archive::ReadLine)
        .def("read_nblock", &Archive::ReadNBlock)
        .def("read_rlblock", &Archive::ReadRLBLOCK)
        .def("read_keyopt_line", &Archive::ReadKEYOPTLine)
        .def("read_et_line", &Archive::ReadETLine)
        .def("read_etblock", &Archive::ReadETBlock)
        .def("read_cmblock", &Archive::ReadCMBlock)
        .def("read_eblock", &Archive::ReadEBlock);
}
