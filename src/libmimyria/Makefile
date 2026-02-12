
# --- optional packages with warnings ---
PKG_CONFIG ?= pkg-config

ifeq ($(shell command -v $(PKG_CONFIG) > /dev/null 2>&1 && echo yes || echo no), yes)

  #liblzma (optional)
  ifneq ($(shell $(PKG_CONFIG) --exists liblzma && echo yes), yes)
    $(warning Optional Package 'liblzma' not found; reading of xz trajectory files will be disabled)
    CXXFLAGS += -DHAVE_LZMA=0
  else 
    CXXFLAGS += $(shell $(PKG_CONFIG) --cflags liblzma) -DHAVE_LZMA=1
    LDLIBS += $(shell $(PKG_CONFIG) --libs liblzma)
  endif

  #libzstd (optional)
  ifneq ($(shell $(PKG_CONFIG) --exists libzstd && echo yes), yes)
    $(warning Optional Package 'liblzma' not found; reading of zstd trajectory files will be disabled)
    CXXFLAGS += -DHAVE_ZSTD=0
  else 
    CXXFLAGS += $(shell $(PKG_CONFIG) --cflags libzstd) -DHAVE_ZSTD=1
    LDLIBS += $(shell $(PKG_CONFIG) --libs libzstd)
  endif

else
	$(warning 'pkg-config' not found; optional packages libzstd and liblzma disabled)
    CXXFLAGS += -DHAVE_ZSTD=0 -DHAVE_LZMA=0
endif


# -- non-optional libraries
override LDLIBS += -lfftw3_omp -lfftw3 -ldl


SRCDIRS := $(wildcard */)
SRC := $(wildcard *.cpp)
SRC += $(foreach I, $(SRCDIRS), $(wildcard $I/*.cpp))
$(info $(SRC))


OBJS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SRC))
DEPS := $(OBJS:.o=.d)

PCH := pch.hpp
PCH_GCH := pch.hpp.gch

CXXFLAGS += -fPIC
LDFLAGS += -shared 

TARGET := $(BUILD_DIR)/libmimyria.so

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LDLIBS)


$(BUILD_DIR)/%.o: %.cpp $(PCH_GCH) | $(BUILD_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -include $(PCH) -c $< -o $@


$(PCH_GCH): $(PCH) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -x c++-header -o $@ $<


$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)


-include $(DEPS)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET) $(PCH_GCH)

.PHONY: all clean
