#include <algorithm>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "array_support.h"

using namespace nb::literals;

/* We are on Windows */
#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#define strtok_r strtok_s
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define NNUM_RESERVE 16384
#define ENUM_RESERVE 16384

static const double DIV_OF_TEN[] = {
    1.0e-0,  1.0e-1,  1.0e-2,  1.0e-3,  1.0e-4,  1.0e-5,  1.0e-6,
    1.0e-7,  1.0e-8,  1.0e-9,  1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13,
    1.0e-14, 1.0e-15, 1.0e-16, 1.0e-17, 1.0e-18, 1.0e-19, 1.0e-20,
    1.0e-21, 1.0e-22, 1.0e-23, 1.0e-24, 1.0e-25, 1.0e-26, 1.0e-27,
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
    fileHandle = CreateFile(filename, GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fileHandle == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("Error opening file");
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(fileHandle, &fileSize)) {
      CloseHandle(fileHandle);
      throw std::runtime_error("Error getting file size");
    }

    size = static_cast<size_t>(fileSize.QuadPart);
    mapHandle =
        CreateFileMapping(fileHandle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapHandle == nullptr) {
      CloseHandle(fileHandle);
      throw std::runtime_error("Error creating file mapping");
    }

    start = static_cast<char *>(
        MapViewOfFile(mapHandle, FILE_MAP_READ, 0, 0, size));
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
    start =
        static_cast<char *>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
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

// Fast ASCI string to integer
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

// Reads various ansys float formats in the form of
// "3.7826539829200E+00"
// "1.0000000000000E-001"
// "        -6.01203 "
//
// fltsz : Number of characters to read in a floating point number
static inline int ans_strtod(char *raw, int fltsz,
                             std::vector<double> &node_vec) {
  char *end = raw + fltsz;
  double sign = 1;

  // skip whitespace
  while (raw < end) {
    if (*raw != ' ') {
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
  // std::cout << val << std::endl;
  if (sign == -1) {
    node_vec.push_back(-val);
  } else {
    node_vec.push_back(val);
  }

  return 0;
}

struct NodeSection {
  NDArray<int, 1> nnum;
  NDArray<double, 2> nodes;
  NDArray<int, 1> tc;
  NDArray<int, 1> rc;

  // Default constructor
  NodeSection() {}

  NodeSection(std::vector<int> nnum_vec, std::vector<double> nodes_vec,
              std::vector<int> tc_vec, std::vector<int> rc_vec) {
    // Assuming the size of the vectors determines the shape of the arrays
    std::array<int, 1> nnum_shape = {static_cast<int>(nnum_vec.size())};
    std::array<int, 2> nodes_shape = {static_cast<int>(nnum_vec.size()), 3};

    // Wrap vectors as NDArrays
    nnum = WrapVectorAsNDArray(std::move(nnum_vec), nnum_shape);
    nodes = WrapVectorAsNDArray(std::move(nodes_vec), nodes_shape);
    tc = WrapVectorAsNDArray(std::move(tc_vec), nnum_shape);
    rc = WrapVectorAsNDArray(std::move(rc_vec), nnum_shape);
  }

  std::string ToString() const {
    std::ostringstream oss;
    int num_nodes = nnum.size();
    if (num_nodes > 1) {
      oss << "NodeSection containing " << num_nodes << " nodes\n\n";
    } else {
      oss << "NodeSection containing " << num_nodes << " node\n\n";
    }

    // Header
    oss << "|  NID  |       X       |       Y       |       Z       |   tc   | "
           "  rc   |\n";
    oss << "|-------|---------------|---------------|---------------|--------|-"
           "-------|\n";

    // Output first 3 nodes in Fortran-like format
    for (int i = 0; i < std::min(10, num_nodes); ++i) {
      oss << std::setw(8) << nnum(i) << " " // Node ID (nnum)
          << std::setw(15) << std::scientific << std::setprecision(8)
          << nodes(i, 0) << " " // X
          << std::setw(15) << std::scientific << std::setprecision(8)
          << nodes(i, 1) << " " // Y
          << std::setw(15) << std::scientific << std::setprecision(8)
          << nodes(i, 2) << " "             // Z
          << std::setw(8) << tc(i) << " "   // tc
          << std::setw(8) << rc(i) << "\n"; // rc
    }

    if (num_nodes > 10) {
      oss << "..." << std::endl;
    }

    return oss.str();
  }
};

struct ElementSection {
  NDArray<int, 1> eid;
  NDArray<int, 1> pid;
  NDArray<int, 1> node_ids;
  NDArray<int, 1> node_id_offsets;
  std::string name = "ElementSection";

  ElementSection() {}

  ElementSection(std::vector<int> eid_vec, std::vector<int> pid_vec,
                 std::vector<int> node_ids_vec,
                 std::vector<int> node_id_offsets_vec) {
    std::array<int, 1> nel_shape = {static_cast<int>(eid_vec.size())};
    std::array<int, 1> node_ids_shape = {
        static_cast<int>(node_id_offsets_vec.back())};
    std::array<int, 1> node_ids_offsets_shape = {
        static_cast<int>(eid_vec.size() + 1)};

    eid = WrapVectorAsNDArray(std::move(eid_vec), nel_shape);
    pid = WrapVectorAsNDArray(std::move(pid_vec), nel_shape);
    node_ids = WrapVectorAsNDArray(std::move(node_ids_vec), node_ids_shape);
    node_id_offsets = WrapVectorAsNDArray(std::move(node_id_offsets_vec),
                                          node_ids_offsets_shape);
  }

  std::string ToString() const {
    std::ostringstream oss;
    int num_elements = eid.size();

    if (num_elements > 1) {
      oss << name << " containing " << num_elements << " elements\n\n";
    } else {
      oss << name << " containing " << num_elements << " element\n\n";
    }

    // Header
    oss << "|  EID  |  PID  |  N1   |  N2   |  N3   |  N4   |  N5   |  N6   |  "
           "N7   |  N8   |\n";
    oss << "|-------|-------|-------|-------|-------|-------|-------|-------|--"
           "-----|-------|\n";

    // Output first 10 elements (or less) in Fortran-like format
    for (int i = 0; i < std::min(10, num_elements); ++i) {
      // Print element ID and part ID
      oss << std::setw(8) << eid(i) << ""  // EID
          << std::setw(8) << pid(i) << ""; // PID

      // Retrieve node IDs associated with this element
      int start = node_id_offsets(i);
      int end = node_id_offsets(i + 1);

      // Print node IDs for the element
      for (int j = start; j < end; ++j) {
        oss << std::setw(8) << node_ids(j) << ""; // NODE_ID
      }

      oss << "\n";
    }

    // Add "..." if there are more than 10 elements
    if (num_elements > 10) {
      oss << "...\n";
    }

    return oss.str();
  }
};

struct ElementSolidSection : public ElementSection {
  ElementSolidSection() : ElementSection() {}

  ElementSolidSection(std::vector<int> eid_vec, std::vector<int> pid_vec,
                      std::vector<int> node_ids_vec,
                      std::vector<int> node_id_offsets_vec)
      : ElementSection(eid_vec, pid_vec, node_ids_vec, node_id_offsets_vec) {
    name = "ElementSolidSection";
  }
};

struct ElementShellSection : public ElementSection {
  ElementShellSection() : ElementSection() {}

  ElementShellSection(std::vector<int> eid_vec, std::vector<int> pid_vec,
                      std::vector<int> node_ids_vec,
                      std::vector<int> node_id_offsets_vec)
      : ElementSection(eid_vec, pid_vec, node_ids_vec, node_id_offsets_vec) {
    name = "ElementShellSection";
  }
};

class Deck {

private:
  bool debug;
  std::string filename;
  MemoryMappedFile memmap;

public:
  std::vector<NodeSection> node_sections;
  std::vector<ElementSolidSection> element_solid_sections;
  std::vector<ElementShellSection> element_shell_sections;

  Deck(const std::string &fname) : filename(fname), memmap(fname.c_str()) {

    // Likely bogus leak warnings. See:
    // https://nanobind.readthedocs.io/en/latest/faq.html#why-am-i-getting-errors-about-leaked-functions-and-types
    // nb::set_leak_warnings(false);
  }

  ~Deck() { memmap.close_file(); }

  // *NODE NID X Y Z TC RC
  // Where TC and RC are translational and rotational constraints:
  // TC Translational Constraint:
  //  EQ.0: no constraints,
  //  EQ.1: constrained x displacement,
  //  EQ.2: constrained y displacement,
  //  EQ.3: constrained z displacement,
  //  EQ.4: constrained x and y displacements,
  //  EQ.5: constrained y and z displacements,
  //  EQ.6: constrained z and x displacements,
  //  EQ.7: constrained x, y, and z displacements.
  // RC Rotational constraint:
  //  EQ.0: no constraints,
  //  EQ.1: constrained x rotation,
  //  EQ.2: constrained y rotation,
  //  EQ.3: constrained z rotation,
  //  EQ.4: constrained x and y rotations,
  //  EQ.5: constrained y and z rotations,
  //  EQ.6: constrained z and x rotations,
  //  EQ.7: constrained x, y, and z rotations.
  //
  // Each node ID in each section is unique
  //
  // Example:
  // *NODE
  //        1-2.309401035E+00-2.309401035E+00-2.309401035E+00       0       0
  //        2-2.039600611E+00-2.039600611E+00-2.039600611E+00       0       0
  void ReadNodeSection() {
    // Assumes that we have already read *NODE and are on the start of the
    // node information

    // Since we don't know the total number of nodes, we'll use vectors here,
    // even though a malloc would be more efficient. Seems they don't store the
    // total number of nodes per section.
    std::vector<int> nnum;
    nnum.reserve(NNUM_RESERVE);
    std::vector<double> nodes;
    nodes.reserve(NNUM_RESERVE * 3);

    std::vector<int> tc;
    tc.reserve(NNUM_RESERVE);

    std::vector<int> rc;
    rc.reserve(NNUM_RESERVE);

    int index = 0;
    while (memmap[0] != '*') {

      // skip comments
      if (memmap[0] == '$') {
        memmap.seek_eol();
      }

      // Read node num (assumes first 8 char)
      nnum.push_back(fast_atoi(memmap.current, 8));
      memmap += 8;

      // next three are always node coordinates in the format of F12.9
      // which comes to 16 characters total
      ans_strtod(memmap.current, 16, nodes);
      memmap += 16;
      ans_strtod(memmap.current, 16, nodes);
      memmap += 16;
      ans_strtod(memmap.current, 16, nodes);
      memmap += 16;

      // constraints
      tc.push_back(fast_atoi(memmap.current, 8));
      memmap += 8;
      rc.push_back(fast_atoi(memmap.current, 8));
      memmap += 8;

      // skip remainder of the line
      memmap.seek_eol();
    }

    NodeSection *node_section = new NodeSection(nnum, nodes, tc, rc);
    node_sections.push_back(*node_section);

    return;
  }

  template <typename T> T ReadElementSection(int num_nodes) {
    std::vector<int> eid;
    eid.reserve(ENUM_RESERVE);

    std::vector<int> pid;
    pid.reserve(ENUM_RESERVE);

    std::vector<int> node_ids;
    node_ids.reserve(ENUM_RESERVE * 20); // using 20 as an upper guess

    std::vector<int> node_id_offsets;
    node_id_offsets.reserve(ENUM_RESERVE);

    int offset = 0;
    node_id_offsets.push_back(0);
    while (memmap[0] != '*') {
      if (memmap[0] == '$') {
        memmap.seek_eol();
        continue;
      }

      eid.push_back(fast_atoi(memmap.current, 8));
      memmap += 8;
      pid.push_back(fast_atoi(memmap.current, 8));
      memmap += 8;

      // Read the specified number of nodes
      for (int i = 0; i < num_nodes; i++) {
        node_ids.push_back(fast_atoi(memmap.current, 8));
        memmap += 8;
      }

      node_id_offsets.push_back(node_ids.size());

      memmap.seek_eol();
    }

    T *element_section = new T(eid, pid, node_ids, node_id_offsets);
    return *element_section;
  }

  // Read the section following the *ELEMENT_SECTION command
  // EID PID NODE0 NODE1 ... NODE_N
  // where:
  //   EID: Element ID
  //   PID: Part ID
  //   NODE0: Index of node0
  // ...
  //
  // Example
  // *ELEMENT_SOLID
  //       1       1       1       2       6       5      17      18      22 21
  //       2       1       2       3       7       6      18      19      23 22
  void ReadElementSolidSection() {
    element_solid_sections.push_back(
        ReadElementSection<ElementSolidSection>(8));
  }

  void ReadElementShellSection() {
    element_shell_sections.push_back(
        ReadElementSection<ElementShellSection>(4));
  }

  int ReadLine() { return memmap.read_line(); }
};

NB_MODULE(_deck, m) {
  nb::class_<NodeSection>(m, "NodeSection")
      .def(nb::init())
      .def("__repr__", &NodeSection::ToString)
      .def_ro("nodes", &NodeSection::nodes, nb::rv_policy::automatic)
      .def_ro("nnum", &NodeSection::nnum, nb::rv_policy::automatic)
      .def_ro("tc", &NodeSection::tc, nb::rv_policy::automatic)
      .def_ro("rc", &NodeSection::rc, nb::rv_policy::automatic);

  nb::class_<ElementSolidSection>(m, "ElementSolidSection")
      .def(nb::init())
      .def("__repr__", &ElementSolidSection::ToString)
      .def_ro("eid", &ElementSolidSection::eid, nb::rv_policy::automatic)
      .def_ro("pid", &ElementSolidSection::pid, nb::rv_policy::automatic)
      .def_ro("node_ids", &ElementSolidSection::node_ids,
              nb::rv_policy::automatic)
      .def_ro("node_id_offsets", &ElementSolidSection::node_id_offsets,
              nb::rv_policy::automatic);

  nb::class_<ElementShellSection>(m, "ElementShellSection")
      .def(nb::init())
      .def("__repr__", &ElementShellSection::ToString)
      .def_ro("eid", &ElementShellSection::eid, nb::rv_policy::automatic)
      .def_ro("pid", &ElementShellSection::pid, nb::rv_policy::automatic)
      .def_ro("node_ids", &ElementShellSection::node_ids,
              nb::rv_policy::automatic)
      .def_ro("node_id_offsets", &ElementShellSection::node_id_offsets,
              nb::rv_policy::automatic);

  nb::class_<Deck>(m, "Deck")
      .def(nb::init<const std::string &>(), "fname"_a, "A LS-DYNA deck.")
      .def_ro("node_sections", &Deck::node_sections)
      .def_ro("element_solid_sections", &Deck::element_solid_sections)
      .def_ro("element_shell_sections", &Deck::element_shell_sections)
      .def("read_line", &Deck::ReadLine)
      .def("read_element_solid_section", &Deck::ReadElementSolidSection)
      .def("read_element_shell_section", &Deck::ReadElementShellSection)
      .def("read_node_section", &Deck::ReadNodeSection);
}
