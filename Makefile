export CC ?= gcc
export CXX ?= g++
export LD = $(CXX)

export CFLAGS ?= -O3 -march=native -D_NDEBUG -fopenmp -g
export CXXFLAGS ?= $(CFLAGS) -std=c++23 -DUSE_GNU -Wall -Wno-ignored-attributes
export DEPFLAGS ?= -MMD -MP
export LDFLAGS := $(CFLAGS)
export LDLIBS := 


SRC_DIR := src
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRC_DIR := $(ROOT)/src
BUILD_DIR ?= $(ROOT)/build

INC := -I$(realpath src/external-libs) 

CXXFLAGS += $(INC)

# Targets:
SUBDIRS := src/libmimyria src/mimyria
LIBMIMYRIA := $(BUILD_DIR)/libmimyria/libmimyria.so
MIMYRIA := $(BUILD_DIR)/mimyria/mimyria

TARGETS := $(LIBMIMYRIA) $(MIMYRIA)

.PHONY: all $(TARGETS) clean

all: $(TARGETS) 

$(LIBMIMYRIA): 
	$(MAKE) -C src/libmimyria BUILD_DIR=$(BUILD_DIR)/libmimyria

$(MIMYRIA): $(LIBMIMYRIA) 
	$(MAKE) -C src/mimyria \
		BUILD_DIR=$(BUILD_DIR)/mimyria \
		CXXFLAGS+='-I$(ROOT)/src/libmimyria' \
		LDFLAGS+='-Wl,-rpath,$(BUILD_DIR)/libmimyria' \
		LDLIBS+='-L$(BUILD_DIR)/libmimyria -lmimyria'


clean:
	@for d in $(SUBDIRS); do \
		$(MAKE) -C $$d clean || exit $$?; \
	done
	@rm -rf "$(BUILD_DIR)"


