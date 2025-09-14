CXX ?= g++
CXXFLAGS := -O3 -march=native -fopenmp -std=c++17
INCLUDES := -I.

ifeq ($(SCHED_RUNTIME),1)
  CXXFLAGS += -DSCHED_RUNTIME
endif

ifeq ($(USE_MKL),1)
  CXXFLAGS += -DUSE_MKL
  ifneq ($(MKLROOT),)
    INCLUDES += -I$(MKLROOT)/include
    LDLIBS  := -L$(MKLROOT)/lib -Wl,--no-as-needed \
               -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -ldl -fopenmp
  else
    LDLIBS  := -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -ldl -fopenmp
  endif
endif

# Include ALL sources referenced by main.cpp
SRCS := main.cpp dgemm_naive.cpp dgemm_blocked.cpp attention_via_dgemm.cpp attention_direct.cpp
OBJS := $(SRCS:.cpp=.o)
BIN := hw_driver

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(BIN) $(OBJS)
