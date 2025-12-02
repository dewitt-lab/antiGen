# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O3 -march=native
DEBUG_FLAGS := -g -O0 -DDEBUG

# Directories
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
BIN_DIR := bin
TEST_DIR := tests

# LibTorch paths (update these for your system)
LIBTORCH_PATH := /path/to/libtorch
TORCH_INCLUDE := $(LIBTORCH_PATH)/include
TORCH_LIB := $(LIBTORCH_PATH)/lib

# Include paths
INCLUDES := -I$(INC_DIR) -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include

# Library paths and libraries
LDFLAGS := -L$(TORCH_LIB) -Wl,-rpath,$(TORCH_LIB)
LDLIBS := -ltorch -ltorch_cpu -lc10

# CUDA support (uncomment if using GPU)
# LDLIBS += -ltorch_cuda -lc10_cuda

# Find source files
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
TEST_SOURCES := $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJECTS := $(TEST_SOURCES:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Main target
TARGET := $(BIN_DIR)/fitness-inference

# Default target
all: directories $(TARGET)

# Create directories
directories:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Build main executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)
	@echo "Build complete: $(TARGET)"

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile test files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Debug build
debug: CXXFLAGS := -std=c++17 -Wall -Wextra $(DEBUG_FLAGS)
debug: clean directories $(TARGET)

# Tests
tests: directories $(TEST_OBJECTS) $(filter-out $(BUILD_DIR)/main.o, $(OBJECTS))
	$(CXX) $(CXXFLAGS) $^ -o $(BIN_DIR)/run_tests $(LDFLAGS) $(LDLIBS)
	@echo "Test build complete: $(BIN_DIR)/run_tests"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Clean and rebuild
rebuild: clean all

# Run the program
run: $(TARGET)
	./$(TARGET)

# Install dependencies (example)
install-deps:
	@echo "Download LibTorch from https://pytorch.org/get-started/locally/"
	@echo "Extract to a location and update LIBTORCH_PATH in Makefile"

# Show help
help:
	@echo "Fitness Inference - Makefile targets:"
	@echo "  make              - Build the project"
	@echo "  make debug        - Build with debug flags"
	@echo "  make tests        - Build tests"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make rebuild      - Clean and rebuild"
	@echo "  make run          - Build and run the program"
	@echo "  make install-deps - Show dependency installation instructions"

# Phony targets
.PHONY: all directories debug tests clean rebuild run install-deps help