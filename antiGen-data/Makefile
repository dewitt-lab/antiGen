NVCC = nvcc
CXXFLAGS = -std=c++17 -O2
INCLUDES = -Iinclude
BUILDDIR = build

SRC_CPP = src/main.cpp src/read_fasta.cpp src/cache.cpp
SRC_CU  = src/attach.cu
OBJ_CPP = $(patsubst src/%.cpp,$(BUILDDIR)/%.o,$(SRC_CPP))
OBJ_CU  = $(patsubst src/%.cu,$(BUILDDIR)/%.o,$(SRC_CU))
OBJ     = $(OBJ_CPP) $(OBJ_CU)

antiGen-data: $(OBJ)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJ)

$(BUILDDIR)/%.o: src/%.cpp | $(BUILDDIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -x cu -c $< -o $@

$(BUILDDIR)/%.o: src/%.cu | $(BUILDDIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) antiGen-data
