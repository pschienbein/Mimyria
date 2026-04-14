

SRCDIRS := $(wildcard */)
SRC := $(wildcard *.cpp)
SRC += $(foreach I, $(SRCDIRS), $(wildcard $I/*.cpp))
$(info $(SRC))


OBJS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SRC))
DEPS := $(OBJS:.o=.d)

$(info $(OBJS))

PCH := pch.hpp
PCH_GCH := pch.hpp.gch

# -- non-optional libraries
override LDLIBS += -lfftw3_omp -lfftw3 -ldl

TARGET := $(BUILD_DIR)/mimyria

$(TARGET): git_info.hpp $(OBJS) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LDLIBS)


$(BUILD_DIR)/%.o: %.cpp $(PCH_GCH) | $(BUILD_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -include $(PCH) -c $< -o $@


$(PCH_GCH): $(PCH) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ -c $<


$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)


git_info.hpp: FORCE
	@{\
		echo '#pragma once'; \
		echo '#define GIT_COMMIT_HASH "'$$(git rev-parse HEAD 2> /dev/null || echo unknown)'"'; \
		echo '#define GIT_COMMIT_SHORT "'$$(git rev-parse --short HEAD 2> /dev/null || echo unknown)'"'; \
		echo '#define GIT_BRANCH "'$$(git rev-parse --abbrev-ref HEAD 2> /dev/null || echo unknown)'"'; \
		echo '#define GIT_DESCRIBE "'$$(git describe --tags --dirty --always 2> /dev/null || echo unknown)'"'; \
	} > $@.tmp
	@cmp -s $@.tmp $@ 2> /dev/null && rm -f $@.tmp || mv $@.tmp $@


-include $(DEPS)

clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d $(TARGET)

.PHONY: all clean FORCE
