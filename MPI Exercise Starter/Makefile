# to compile the code
# module load PrgEnv-gnu 
CXX = CC

# Compiler flags
CXXFLAGS = -O3 -std=c++11 -fopenmp -Wall

# Executable name
TARGET = cov_mpi

# Source files
SRC = cov_mpi.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o
