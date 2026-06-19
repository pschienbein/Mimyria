
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
    $(warning Optional Package 'libzstd' not found; reading of zstd trajectory files will be disabled)
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

$(TARGET): mimyria_git_info.hpp $(OBJS) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LDLIBS)


$(BUILD_DIR)/%.o: %.cpp $(PCH_GCH) | $(BUILD_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -include $(PCH) -c $< -o $@


$(PCH_GCH): $(PCH) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -x c++-header -o $@ $<


$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)


mimyria_git_info.hpp: FORCE
	@{\
        echo '#pragma once'; \
        echo ''; \
        echo 'namespace libmimyria::git_info {'; \
        echo ''; \
        echo 'inline constexpr const char* commit_hash  = "'$$(git rev-parse HEAD 2>/dev/null || echo unknown)'";'; \
        echo 'inline constexpr const char* commit_short = "'$$(git rev-parse --short HEAD 2>/dev/null || echo unknown)'";'; \
        echo 'inline constexpr const char* branch       = "'$$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)'";'; \
        echo 'inline constexpr const char* describe     = "'$$(git describe --tags --dirty --always 2>/dev/null || echo unknown)'";'; \
		echo 'inline constexpr const char* banner = "This is mimyria, '$$(git describe --tags --dirty --always 2>/dev/null || echo unknown)' (Commit '$$(git rev-parse --short HEAD 2>/dev/null || echo unknown)' @ '$$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)' branch)";'; \
        echo ''; \
        echo '} // namespace libmimyria::git_info'; \
	} > $@.tmp
	@cmp -s $@.tmp $@ 2> /dev/null && rm -f $@.tmp || mv $@.tmp $@


-include $(DEPS)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET) $(PCH_GCH)

.PHONY: all clean FORCE
