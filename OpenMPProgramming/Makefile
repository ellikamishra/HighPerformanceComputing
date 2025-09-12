CXX ?= g++
CXXFLAGS := -O3 -march=native -fopenmp -std=c++17
INCLUDES := -I.

ifeq ($(USE_MKL),1)
  CXXFLAGS += -DUSE_MKL
  ifneq ($(MKLROOT),)
    INCLUDES += -I$(MKLROOT)/include
    LDLIBS  := -L$(MKLROOT)/lib -Wl,--no-as-needed \\
               -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -ldl -fopenmp
  else
    LDLIBS  := -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -ldl -fopenmp
  endif
endif

SRCS := main.cpp dgemm_naive.cpp 
OBJS := $(SRCS:.cpp=.o)
BIN := hw_driver

all: $(BIN)
$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(BIN) $(OBJS)
