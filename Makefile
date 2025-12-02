CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=1
INCLUDES = -Iinclude
LDFLAGS = 

# LibTorch paths
LIBTORCH_PATH = /opt/miniconda3/lib/python3.13/site-packages/torch
INCLUDES += -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
LDFLAGS += -L$(LIBTORCH_PATH)/lib -ltorch -lc10 -ltorch_cpu
LDFLAGS += -Wl,-rpath,$(LIBTORCH_PATH)/lib

# For macOS, use:
# LDFLAGS += -Wl,-rpath,@executable_path/../lib

TARGET = fitness-inference

# Source and object file directories
SRC_DIR = src
OBJ_DIR = obj
INC_DIR = include

# Source files
SOURCES = $(SRC_DIR)/main.cpp $(SRC_DIR)/tree.cpp $(SRC_DIR)/fitness_transformer.cpp
OBJECTS = $(OBJ_DIR)/main.o $(OBJ_DIR)/tree.o $(OBJ_DIR)/fitness_transformer.o

# Header files
HEADERS = $(INC_DIR)/tree.h $(INC_DIR)/fitness_transformer.h

# Default target
all: $(OBJ_DIR) $(TARGET)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

rebuild: clean all

run: $(TARGET)
	./$(TARGET) ncov_global.json NC_045512.2.fasta

.PHONY: all clean rebuild run